import argparse
from dataclasses import dataclass
import time
import numpy as np

import spatialmath as sm
import spatialmath.base as smb
from loop_rate_limiters import RateLimiter
import rerun as rr

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_


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

    def copy(self) -> "Pose":
        return Pose(
            x=self.x,
            y=self.y,
            z=self.z,
            w=self.w,
            rx=self.rx,
            ry=self.ry,
            rz=self.rz,
        )

    def update_from_se3(self, transform: sm.SE3):
        quat = smb.r2q(transform.R, order="sxyz")
        self.x = float(transform.t[0])
        self.y = float(transform.t[1])
        self.z = float(transform.t[2])
        self.w = float(quat[0])
        self.rx = float(quat[1])
        self.ry = float(quat[2])
        self.rz = float(quat[3])




@dataclass

class PIDGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.0
    output_limit: float = 1.0


class PIDAxis:
    def __init__(self, gains: PIDGains, name: str, log: bool = False):
        self.gains = gains
        self.integral = 0.0
        self.previous_error: float | None = None
        self.name = name
        self.log = log

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        if self.gains.integral_limit > 0.0:
            self.integral += error * dt
            self.integral = np.clip(
                self.integral,
                -self.gains.integral_limit,
                self.gains.integral_limit,
            )
        else:
            self.integral = 0.0

        derivative = 0.0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * derivative
        )

        if self.log:
            rr.log(f"/pid/{self.name}/error", rr.Scalars(float(error)))
            rr.log(f"/pid/{self.name}/integral", rr.Scalars(float(self.integral)))
            rr.log(f"/pid/{self.name}/derivative", rr.Scalars(float(derivative)))
            rr.log(f"/pid/{self.name}/output", rr.Scalars(float(output)))
        return float(np.clip(output, -self.gains.output_limit, self.gains.output_limit))


class G1BaseController:
    def __init__(
        self,
        hz: int = 50,
        target_pose_topic="rt/dolly_pose",
        robot_pose_topic="rt/torso_pose",
        robot_state_topic="rt/sportmodestate",
        base_command="rt/run_command/cmd",
        pid_x_gains: PIDGains | None = None,
        pid_y_gains: PIDGains | None = None,
        pid_yaw_gains: PIDGains | None = None,
        log: bool = False,
        enable_x: bool = False,
        enable_y: bool = False,
        enable_yaw: bool = False,
    ):
        self.hz = hz
        self.target_vel = [0.0, 0.0, 0.0]

        self.x_wrp = Pose()
        self.x_wr = Pose()
        self.target_step_test = target_step_test
        self.test_initial_pose: Pose | None = None
        self.test_started_at: float | None = None
        self.test_step_applied = False
        self.enable_x = enable_x
        self.enable_y = enable_y
        self.enable_yaw = enable_yaw
        self.current_vel = np.zeros(3, dtype=float)
        self.command_vel = np.zeros(3, dtype=float)
        self.last_step_time = time.monotonic()

        x_Ku = 30.0
        x_Tu= 1.8

        ## No overshoot rules
        x_kp = 0.2 * x_Ku
        x_ki = 0.4 * x_Ku / x_Tu
        x_kd = 0.066 * x_Ku * x_Tu


        self.pid_x = PIDAxis(
            pid_x_gains or PIDGains(kp=x_kp, ki=x_ki, kd=x_kd, output_limit=0.8, integral_limit=0.8 / x_ki),
            name="x",
            log=log,
        )

        y_Ku = 20.0
        y_Tu= 1.1

        ## No overshoot rules
        y_kp = 0.2 * y_Ku
        y_ki = 0.4 * y_Ku / y_Tu
        y_kd = 0.066 * y_Ku * y_Tu


        self.pid_y = PIDAxis(
            pid_y_gains or PIDGains(kp=y_kp, ki=y_ki, kd = y_kd, output_limit=0.5, integral_limit=0.5/y_ki),
            name="y",
            log=log,
        )
        yaw_Ku = 80.0
        yaw_Tu= 0.8
        # yaw_kd = 0.075 * yaw_Ku * yaw_Tu

        ## No overshoot rules
        yaw_kp = 0.2 * yaw_Ku
        yaw_ki = 0.4 * yaw_Ku / yaw_Tu
        yaw_kd = 0.066 * yaw_Ku * yaw_Tu


        self.pid_yaw = PIDAxis(
            pid_yaw_gains or PIDGains(kp=yaw_kp, ki=yaw_ki, kd=yaw_kd, output_limit=1.0, integral_limit=1.0/ yaw_ki),
            name="yaw",
            log=log,
        )

        self.target_pose_first_frame = False
        self.robot_pose_first_frame = False
        self.robot_state_first_frame = False

        self.command_publisher = ChannelPublisher(base_command, String_)
        self.command_publisher.Init()

        self.target_pose_sub = ChannelSubscriber(target_pose_topic, PoseStamped_)
        self.target_pose_sub.Init(self.target_pose_callback, 1)

        self.robot_pose_sub = ChannelSubscriber(robot_pose_topic, PoseStamped_)
        self.robot_pose_sub.Init(self.robot_pose_callback, 1)

        self.robot_state_sub = ChannelSubscriber(robot_state_topic, SportModeState_)
        self.robot_state_sub.Init(self.robot_state_callback, 1)

        while not self.ready():
            print("Waiting for topics")
            time.sleep(1)

        print("All topics received. Starting controller.")
        if self.target_step_test is not None:
            self.test_started_at = time.monotonic()
            print(
                "Target step test armed: "
                f"x={self.target_step_test.x_step:.3f} m, "
                f"y={self.target_step_test.y_step:.3f} m, "
                f"yaw={self.target_step_test.yaw_step:.3f} rad, "
                f"delay={self.target_step_test.delay_s:.3f} s"
            )

        rate = RateLimiter(hz)
        while True:
            self.step()
            rate.sleep()

    def ready(self):
        return (
            self.robot_state_first_frame
            and self.target_pose_first_frame
            and self.robot_pose_first_frame
        )

    def robot_state_callback(self, msg: SportModeState_):
        first_frame = not self.robot_state_first_frame
        self.current_vel = np.asarray(msg.velocity, dtype=float)[:3]
        if first_frame:
            self.command_vel = self.current_vel.copy()
        self.robot_state_first_frame = True

    def target_pose_callback(self, msg: PoseStamped_):
        self.update_pose_from_msg(msg, self.x_wrp)
        self.target_pose_first_frame = True

    def robot_pose_callback(self, msg: PoseStamped_):
        self.update_pose_from_msg(msg, self.x_wr)
        if self.target_step_test is not None and not self.target_pose_first_frame:
            self.x_wrp = self.x_wr.copy()
            self.test_initial_pose = self.x_wr.copy()
            self.target_pose_first_frame = True
        self.robot_pose_first_frame = True

    def update_pose_from_msg(self, msg: PoseStamped_, pose: Pose):
        pos = msg.pose.position
        quat = msg.pose.orientation

        pose.x = pos.x
        pose.y = pos.y
        pose.z = pos.z

        pose.w = quat.w
        pose.rx = quat.x
        pose.ry = quat.y
        pose.rz = quat.z

    def publish_command(self, vx, vy, omega, dt=None):
        if dt is None:
            dt = 1.0 / self.hz

        target = np.array([vx, vy, omega], dtype=float)

        if not hasattr(self, "command_vel"):
            self.command_vel = np.zeros(3, dtype=float)

        max_acc = np.array([
            0.3,  # max vx acceleration [m/s^2]
            0.2,  # max vy acceleration [m/s^2]
            0.8,  # max omega acceleration [rad/s^2]
        ])

        delta = target - self.command_vel
        max_delta = max_acc * dt

        delta = np.clip(delta, -max_delta, max_delta)

        self.command_vel = target

        data = f"{self.command_vel[0]} {self.command_vel[1]} {self.command_vel[2]} 0.8"
        cmd = String_(data)
        self.command_publisher.Write(cmd)

   
    def step(self):
        # compute and publish command here
        now = time.monotonic()
        dt = now - self.last_step_time
        self.last_step_time = now

        if dt <= 0.0 or dt > 1.0:
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_yaw.reset()
            dt = 1.0 / self.hz

        x_rrp: sm.SE3 = self.x_wr.to_se3().inv() * self.x_wrp.to_se3()

        ex, ey = x_rrp.t[:2]
        rpy = x_rrp.rpy(order="zyx")
        yaw_error = rpy[2]

        vx = self.pid_x.update(ex, dt)
        vy = self.pid_y.update(ey, dt)
        omega = self.pid_yaw.update(yaw_error, dt)
        self.publish_command(vx, vy, omega, dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the G1 base controller. By default this starts a standalone test target: "
            "hold the first received torso pose, then move to the configured relative pose."
        )
    )
    parser.add_argument("--hz", type=int, default=50)
    parser.add_argument("--network-interface", default="lo")
    parser.add_argument("--dds-id", type=int, default=1)
    parser.add_argument("--target-pose-topic", default="rt/dolly_pose")
    parser.add_argument("--robot-pose-topic", default="rt/torso_pose")
    parser.add_argument("--robot-state-topic", default="rt/sportmodestate")
    parser.add_argument("--base-command-topic", default="rt/run_command/cmd")
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a built-in relative target pose instead of subscribing to --target-pose-topic. "
            "Pass --no-test to follow the target pose topic."
        ),
    )
    parser.add_argument("--test-x-step", type=float, default=0.0, help="Relative target x step in meters.")
    parser.add_argument("--test-y-step", type=float, default=0.0, help="Relative target y step in meters.")
    parser.add_argument(
        "--test-yaw-step-deg",
        type=float,
        default=20.0,
        help="Relative target yaw step in degrees.",
    )
    parser.add_argument("--test-delay-s", type=float, default=2.0)
    parser.add_argument("--enable-x", action="store_true")
    parser.add_argument("--enable-y", action="store_true")
    parser.add_argument("--enable-yaw", action="store_true")
    args = parser.parse_args()

    rr.init("g1_base_controller_v5", spawn=True)
    ChannelFactoryInitialize(id=args.dds_id, networkInterface=args.network_interface)

    target_step_test = None
    if args.test:
        target_step_test = TargetStepTest(
            x_step=args.test_x_step,
            y_step=args.test_y_step,
            yaw_step=float(np.deg2rad(args.test_yaw_step_deg)),
            delay_s=args.test_delay_s,
        )

    G1BaseController(
        log=True,
        hz=args.hz,
        target_pose_topic=args.target_pose_topic,
        robot_pose_topic=args.robot_pose_topic,
        robot_state_topic=args.robot_state_topic,
        base_command=args.base_command_topic,
        target_step_test=target_step_test,
        enable_x=args.enable_x,
        enable_y=args.enable_y,
        enable_yaw=args.enable_yaw,
    )
