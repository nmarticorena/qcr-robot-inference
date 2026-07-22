from rs_imle_policy.configs.g1_configs import G1IKConfigSim
from dataclasses import dataclass
import threading
import json
import re
from rs_imle_policy.unitree import G1_29_ArmController
import time
from rs_imle_policy.g1_arm_ik import G1ReducedPinkIK
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from motion_tools.robot_gui import ReRunRobot
import pinocchio as pin
import rerun as rr
from loop_rate_limiters import RateLimiter
import spatialmath as sm
import spatialmath.base as smb

from rs_imle_policy.utils.transforms import se3_from_pos_quat
from rs_imle_policy.inspire import Inspire_Controller_Sim

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_

ChannelFactoryInitialize(id=1, networkInterface="lo")  # dds domain id

# Hardcoded offset of trolly body to handle, currently using the side ones
X_dl = se3_from_pos_quat(np.array([-0.45, 0.13, 0.35]), np.array([1.0, 0.0, 0.0, 0.0]))[0]
X_dr = se3_from_pos_quat(np.array([-0.45, -0.13, 0.35]), np.array([1.0, 0.0, 0.0, 0.0]))[0]
# X_dl = X_dl * sm.SE3.Rx(np.pi / 2)  # Rotate left handle to align with dolly
# X_dr = X_dr * sm.SE3.Rx(-np.pi / 2)  # Rotate right handle to align with dolly


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def to_se3(self) -> sm.SE3:
        # Convert quaternion to rotation matrix
        quat = [self.w, self.rx, self.ry, self.rz]
        R = smb.q2r(quat, order="sxyz")  # Convert quaternion to rotation matrix
        t = np.array([self.x, self.y, self.z])
        T = sm.SE3(t)
        T.A[:3, :3] = R
        return T

    def to_rerun(self) -> tuple[list[float], list[float]]:
        return [self.x, self.y, self.z], [self.rx, self.ry, self.rz, self.w]


dolly_pose_lock = threading.Lock()
robot_pose_lock = threading.Lock()
X_wd = Pose()
X_wr = Pose()


class FrontVideoSubscriber:
    """Caches the latest RGB frame from the simulator front camera topic."""

    _COMMON_RGB_SHAPES = (
        (1280, 720),
        (640, 480),
        (640, 360),
        (320, 180),
        (1920, 1080),
        (848, 480),
        (800, 600),
    )

    def __init__(self, topic: str = "rt/head_camera/front_video"):
        self._lock = threading.Lock()
        self._rgb = None
        self._time_frame = None
        self._seq = 0
        self._checksum = None
        self._last_debug = 0.0
        self._warned_invalid = False
        self._subscriber = ChannelSubscriber(topic, Go2FrontVideoData_)
        self._subscriber.Init(self._handler, 0)

    def get(self):
        with self._lock:
            return self._rgb, self._seq, self._time_frame, self._checksum

    def _handler(self, msg: Go2FrontVideoData_):
        try:
            rgb = self._decode_rgb(msg)
        except ValueError as exc:
            now = time.monotonic()
            if not self._warned_invalid or now - self._last_debug >= 2.0:
                print(
                    "Skipping front camera frame: "
                    f"{exc}; video720p={self._payload_len(msg.video720p)} "
                    f"video180p={self._payload_len(msg.video180p)} "
                    f"time_frame={msg.time_frame}"
                )
                self._warned_invalid = True
                self._last_debug = now
            return

        checksum = self._rgb_checksum(rgb)
        now = time.monotonic()
        with self._lock:
            self._seq += 1
            self._rgb = rgb
            self._time_frame = msg.time_frame
            self._checksum = checksum
            seq = self._seq
            should_print = now - self._last_debug >= 2.0
            if should_print:
                self._last_debug = now

        if should_print:
            print(f"Front camera frame {seq}: shape={rgb.shape} time_frame={msg.time_frame} checksum={checksum}")

    def _decode_rgb(self, msg: Go2FrontVideoData_) -> np.ndarray:
        rgb_payload = msg.video720p
        if isinstance(rgb_payload, (bytes, bytearray, memoryview)):
            rgb_flat = np.frombuffer(rgb_payload, dtype=np.uint8)
        else:
            rgb_flat = np.asarray(rgb_payload, dtype=np.uint8)

        if rgb_flat.size == 0:
            raise ValueError("empty video720p payload")
        if rgb_flat.size % 3 != 0:
            raise ValueError(f"RGB payload size {rgb_flat.size} is not divisible by 3")

        metadata = self._decode_metadata(msg.video180p)
        width, height = self._rgb_shape(metadata, rgb_flat.size)
        expected_size = width * height * 3
        if rgb_flat.size != expected_size:
            raise ValueError(f"RGB payload size {rgb_flat.size} does not match {width}x{height} RGB8")

        return rgb_flat.reshape((height, width, 3)).copy()

    @staticmethod
    def _decode_metadata(metadata_payload) -> str:
        if not metadata_payload:
            return ""
        if isinstance(metadata_payload, str):
            return metadata_payload.strip()
        if isinstance(metadata_payload, (bytes, bytearray, memoryview)):
            metadata_bytes = bytes(metadata_payload)
        else:
            metadata_bytes = bytes(int(value) & 0xFF for value in metadata_payload)
        metadata_bytes = metadata_bytes.split(b"\x00", 1)[0]
        return metadata_bytes.decode("ascii", errors="ignore").strip()

    @classmethod
    def _rgb_shape(cls, metadata: str, payload_size: int) -> tuple[int, int]:
        width, height = cls._shape_from_metadata(metadata)
        if width is not None and height is not None:
            return width, height

        pixels = payload_size // 3
        for width, height in cls._COMMON_RGB_SHAPES:
            if width * height == pixels:
                return width, height

        if pixels % 720 == 0:
            return pixels // 720, 720

        raise ValueError(f"cannot infer RGB frame dimensions from {payload_size} bytes")

    @classmethod
    def _shape_from_metadata(cls, metadata: str) -> tuple[int | None, int | None]:
        if not metadata:
            return None, None

        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            width = cls._first_int(parsed, ("rgb_width", "color_width", "width", "w"))
            height = cls._first_int(parsed, ("rgb_height", "color_height", "height", "h"))
            if width is not None and height is not None:
                return width, height

            for key in ("rgb", "color", "video720p", "video_720p"):
                nested = parsed.get(key)
                if isinstance(nested, dict):
                    width = cls._first_int(nested, ("width", "w"))
                    height = cls._first_int(nested, ("height", "h"))
                    if width is not None and height is not None:
                        return width, height

        pairs = {
            key.lower(): int(value) for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(\d+)", metadata)
        }
        width = cls._first_int(pairs, ("rgb_width", "color_width", "width", "w"))
        height = cls._first_int(pairs, ("rgb_height", "color_height", "height", "h"))
        return width, height

    @staticmethod
    def _first_int(data: dict, keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _payload_len(payload) -> int:
        if payload is None:
            return 0
        try:
            return len(payload)
        except TypeError:
            return 0

    @staticmethod
    def _rgb_checksum(rgb: np.ndarray) -> int:
        flat = rgb.reshape(-1)
        stride = max(1, flat.size // 4096)
        return int(flat[::stride].astype(np.uint64).sum() % (1 << 32))


def dolly_pose_handler(msg: PoseStamped_):
    global X_wd

    pos = msg.pose.position
    quat = msg.pose.orientation

    with dolly_pose_lock:
        X_wd.x = pos.x
        X_wd.y = pos.y
        X_wd.z = pos.z
        X_wd.w = quat.w
        X_wd.rx = quat.x
        X_wd.ry = quat.y
        X_wd.rz = quat.z


def robot_pose_handler(msg: PoseStamped_):
    global X_wr

    pos = msg.pose.position
    quat = msg.pose.orientation

    with robot_pose_lock:
        X_wr.x = pos.x
        X_wr.y = pos.y
        X_wr.z = pos.z
        X_wr.w = quat.w
        X_wr.rx = quat.x
        X_wr.ry = quat.y
        X_wr.rz = quat.z


dolly_pose_sub = ChannelSubscriber("rt/dolly_pose", PoseStamped_)
dolly_pose_sub.Init(dolly_pose_handler, 0)

robot_pose_sub = ChannelSubscriber("rt/torso_pose", PoseStamped_)
robot_pose_sub.Init(robot_pose_handler, 0)

front_video_sub = FrontVideoSubscriber()


def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)


reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
reset_pose_publisher.Init()
publish_reset_category(1, reset_pose_publisher)


rec = rr.RecordingStream("g1_arm_controller_test")
controller = G1_29_ArmController(motion_mode=True, simulation_mode=False)
hands = Inspire_Controller_Sim(simulation_mode=True)
frecuecy = 200
rec.spawn()

ik = G1ReducedPinkIK(
    config=G1IKConfigSim(),
    spawn_visualizer=False,
)


def clipping(distance):
    distance -= 0.45
    return 1 - 1 / (1 + np.exp(-15 * (distance - 0.5)))


targets = ik.get_targets_from_configuration()
ik.set_targets(targets.left, targets.right)

robot_sol = ReRunRobot.g1_debug(rec, target_frame="pelvis")
robot_gui = ReRunRobot.g1(rec, target_frame="world")
robot_sol.apply_color([0, 1, 0, 0.5])


ti = time.time()
while time.time() - ti < 0.05:
    pass

rate = RateLimiter(frecuecy, warn=False)
camera_rate = 0
last_camera_seq = 0
last_camera_report = 0.0
while True:
    target = ik.get_targets()
    robot_pos, robot_quat = X_wr.to_rerun()
    robot_sol.log_transform_named_frames("x_wr", robot_pos, robot_quat, parent_frame="world", child_frame="pelvis")
    dolly_pos, dolly_quat = X_wd.to_rerun()
    robot_sol.log_transform_named_frames("x_wd", dolly_pos, dolly_quat, parent_frame="world", child_frame="dolly")

    x_wdl: sm.SE3 = X_wd.to_se3() * X_dl

    x_wdr: sm.SE3 = X_wd.to_se3() * X_dr

    x_rdl: sm.SE3 = X_wr.to_se3().inv() * x_wdl
    x_rdr: sm.SE3 = X_wr.to_se3().inv() * x_wdr

    robot_sol.log_se3_transform("x_wdl", x_wdl, parent_frame="world")
    robot_sol.log_se3_transform("x_wdr", x_wdr, parent_frame="world")
    # robot_sol.log_transform_named_frames("x_wdl", parent_frame ="world", child_frame="dolly_left", pos=x_wdl.t, quat=x_wdl.R)
    # robot_sol.log_transform_named_frames("x_wdr", parent_frame ="world", child_frame="dolly_right", pos=w_wdr.t, quat=w_wdr.q)

    robot_sol.log_pin_transform("left_ee", target.left)
    robot_sol.log_pin_transform("right_ee", target.right)
    ti = time.time()
    q = controller.get_current_motor_q()
    q_arm = controller.get_current_dual_arm_q()
    q_left_arm = q_arm[:7]
    q_right_arm = q_arm[7:14]

    distance = np.linalg.norm(X_wr.to_se3().t - X_wd.to_se3().t)

    print(distance)
    print(clipping(distance))

    # distance.clip(min=0.01, max=1.0)  # Avoid division by zero and limit the maximum distance

    dq_arm = controller.get_current_dual_arm_dq()
    ik.set_target_smooth(x_rdl, x_rdr, alpha=clipping(distance))
    left_e, right_e = ik.pos_error()

    left_hand_target = np.ones(6) * (0 if left_e < 0.1 else 1)
    right_hand_target = np.ones(6) * (0 if right_e < 0.1 else 1)
    # left_hand_target = np.zeros(6)
    # right_hand_target = np.zeros(6)

    hands.policy_to_hand_command(left_hand_target, right_hand_target)
    ik.configuration.update(q_arm.copy())
    # ik.viz.display(q_arm.copy())
    q_sol = ik.solve(dt=1 / 200, n_steps=1)
    full_q_sol = np.zeros(29)
    full_q_sol[15:29] = q_sol
    robot_sol.log(full_q_sol)
    robot_gui.log(q.copy())

    q_tauff = pin.rnea(
        ik.robot.model,
        ik.robot.data,
        q_sol,
        np.zeros(ik.robot.model.nv),
        np.zeros(ik.robot.model.nv),
    )
    controller.ctrl_dual_arm(q_sol, q_tauff)
    if camera_rate % 1 == 0:  # We record at roughly 200hz
        rgb, camera_seq, time_frame, checksum = front_video_sub.get()
        if rgb is None:
            now = time.monotonic()
            if now - last_camera_report >= 2.0:
                print("Waiting for front camera DDS frame on rt/head_camera/front_video")
                last_camera_report = now
        elif camera_seq != last_camera_seq:
            robot_gui.rec.set_time("camera_frame", sequence=camera_seq)
            robot_gui.rec.log(
                "cameras/head_frame",
                rr.Image(rgb).compress(jpeg_quality=40),
            )
            robot_gui.rec.reset_time()
            print(f"Logged front camera frame {camera_seq}: time_frame={time_frame} checksum={checksum}")
            last_camera_seq = camera_seq
    camera_rate += 1

    rate.sleep()
