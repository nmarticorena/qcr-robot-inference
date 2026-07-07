from rs_imle_policy.configs.g1_configs import PositionBarrierBounds, G1IKConfigSim
from dataclasses import dataclass
import threading
import json
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

from teleimager.image_client import ImageClient

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

img_client = ImageClient("vlu-isaacsim.qut.edu.au", request_bgr=True)

ChannelFactoryInitialize(id=1)#, networkInterface="lo")  # dds domain id

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
        T.A[:3,:3] = R
        return T

    def to_rerun(self) -> tuple[list[float], list[float]]:
        return [self.x, self.y, self.z], [self.rx, self.ry, self.rz, self.w]

dolly_pose_lock = threading.Lock()
robot_pose_lock = threading.Lock()
X_wd = Pose()
X_wr = Pose()


def dolly_pose_handler(msg: String_):
    global X_wd
    try:
        # If your publisher sends JSON, e.g.
        # {"pos":[x,y,z], "quat":[w,x,y,z]}
        data = json.loads(msg.data)
    except json.JSONDecodeError:
        # Fallback: store raw string
        data = msg.data
    assert isinstance(data,dict)

    pos = data.get("position", [0, 0, 0])
    quat = data.get("quaternion_wxyz", [1, 0, 0, 0])  # Default to identity quaternion if not provided
    with dolly_pose_lock:
        X_wd.x = pos[0]
        X_wd.y = pos[1]
        X_wd.z = pos[2]
        X_wd.w = quat[0]
        X_wd.rx = quat[1]
        X_wd.ry = quat[2]
        X_wd.rz = quat[3]

def robot_pose_handler(msg: String_):
    global X_wr
    try:
        # If your publisher sends JSON, e.g.
        # {"pos":[x,y,z], "quat":[w,x,y,z]}
        data = json.loads(msg.data)
    except json.JSONDecodeError:
        # Fallback: store raw string
        data = msg.data
    assert isinstance(data,dict)

    pos = data.get("position", [0, 0, 0])
    quat = data.get("quaternion_wxyz", [1, 0, 0, 0])  # Default to identity quaternion if not provided
    with robot_pose_lock:
        X_wr.x = pos[0]
        X_wr.y = pos[1]
        X_wr.z = pos[2]
        X_wr.w = quat[0]
        X_wr.rx = quat[1]
        X_wr.ry = quat[2]
        X_wr.rz = quat[3]


dolly_pose_sub = ChannelSubscriber("rt/dolly/pose", String_)
dolly_pose_sub.Init(dolly_pose_handler, 10)

robot_pose_sub = ChannelSubscriber("rt/torso/pose", String_)
robot_pose_sub.Init(robot_pose_handler, 10)




def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)


reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
reset_pose_publisher.Init()
publish_reset_category(1, reset_pose_publisher)


rec = rr.RecordingStream("g1_arm_controller_test")
controller = G1_29_ArmController(motion_mode=False, simulation_mode=True)
hands = Inspire_Controller_Sim(simulation_mode=True)
frecuecy = 200
rec.spawn()

ik = G1ReducedPinkIK(
    config = G1IKConfigSim(),
    spawn_visualizer=False,
)


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
while True:
    target = ik.get_targets()
    robot_pos, robot_quat = X_wr.to_rerun()
    robot_sol.log_transform_named_frames("x_wr", robot_pos, robot_quat, parent_frame ="world", child_frame="torso_link")
    dolly_pos, dolly_quat = X_wd.to_rerun()
    robot_sol.log_transform_named_frames("x_wd", dolly_pos, dolly_quat, parent_frame ="world", child_frame="dolly")
    
    x_wdl:sm.SE3 = X_wd.to_se3() * X_dl

    x_wdr:sm.SE3 = X_wd.to_se3() * X_dr

    x_rdl:sm.SE3 = X_wr.to_se3().inv() * x_wdl
    x_rdr:sm.SE3 = X_wr.to_se3().inv() * x_wdr


    robot_sol.log_se3_transform("x_wdl", x_wdl, parent_frame = "world")
    robot_sol.log_se3_transform("x_wdr", x_wdr, parent_frame = "world")
    # robot_sol.log_transform_named_frames("x_wdl", parent_frame ="world", child_frame="dolly_left", pos=x_wdl.t, quat=x_wdl.R)
    # robot_sol.log_transform_named_frames("x_wdr", parent_frame ="world", child_frame="dolly_right", pos=w_wdr.t, quat=w_wdr.q)

    robot_sol.log_pin_transform("left_ee", target.left)
    robot_sol.log_pin_transform("right_ee", target.right)
    ti = time.time()
    q = controller.get_current_motor_q()
    q_arm = controller.get_current_dual_arm_q()
    q_left_arm = q_arm[:7]
    q_right_arm = q_arm[7:14]

    dq_arm = controller.get_current_dual_arm_dq()
    ik.set_targets(x_rdl, x_rdr)
    left_e , right_e = ik.pos_error()

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
    if camera_rate % 20 == 0: # We record at roughly 10hz
        robot_gui.rec.log(
            "cameras/head_frame",
            rr.EncodedImage(
                contents=img_client.get_head_frame().jpg, media_type="image/jpeg"
            ),
        )
    camera_rate += 1

    rate.sleep()
