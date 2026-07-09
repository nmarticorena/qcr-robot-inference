from dataclasses import dataclass
import time
import numpy as np

import spatialmath as sm
import spatialmath.base as smb
from loop_rate_limiters import RateLimiter

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.utils.crc import CRC
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


class G1BaseController:
    def __init__(
        self,
        hz: int = 50,
        target_pose_topic="rt/dolly_pose",
        robot_pose_topic="rt/torso_pose",
        robot_state_topic="rt/sportmodestate",
        base_command="rt/run_command/cmd",
    ):
        self.hz = hz
        self.target_vel = [0.0, 0.0, 0.0]

        self.x_wrp = Pose()
        self.x_wr = Pose()
        self.current_vel = [0.0, 0.0, 0.0]

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
            # print("target ready:", self.target_pose_first_frame, self.target_pose)
            # print("robot ready:", self.robot_pose_first_frame, self.robot_pose)
            # print("state ready:", self.robot_state_first_frame, self.current_vel)
            time.sleep(1)

        print("All topics received. Starting controller.")

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
        self.current_vel = msg.velocity
        self.robot_state_first_frame = True

    def target_pose_callback(self, msg: PoseStamped_):
        self.update_pose_from_msg(msg, self.x_wrp)
        self.target_pose_first_frame = True

    def robot_pose_callback(self, msg: PoseStamped_):
        self.update_pose_from_msg(msg, self.x_wr)
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

    def publish_command(self, vx,vy,omega):
        dt = 1.0 / self.hz

        target = np.array([vx, vy, omega], dtype=float)

        if not hasattr(self, "current_vel"):
            self.current_vel = np.zeros(3)

        max_acc = np.array([
            300,  # max vx acceleration [m/s^2]
            200,  # max vy acceleration [m/s^2]
            100.,  # max omega acceleration [rad/s^2]
        ])

        delta = target - self.current_vel
        max_delta = max_acc * dt

        delta = np.clip(delta, -max_delta, max_delta)

        self.current_vel = self.current_vel + delta

        data = f"{self.current_vel[-1]} {self.current_vel[1]} {self.current_vel[2]} 0.8"
        cmd = String_(data)
        self.command_publisher.Write(cmd)

    def step(self):
        # compute and publish command here
        x_rrp:sm.SE3 = self.x_wr.to_se3().inv() * self.x_wrp.to_se3() # Local pose

        vx, vy = x_rrp.t[:2] * 0.5
        rpy = x_rrp.rpy(order ="zyx")
        omega = rpy[0] * 10
        self.publish_command(vx,vy,omega)


    

if __name__ == "__main__":
    ChannelFactoryInitialize(id = 1, networkInterface="lo")

    base_controller = G1BaseController()
