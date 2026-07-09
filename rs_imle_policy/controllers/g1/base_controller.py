from dataclasses import dataclass
import time

import spatialmath as sm
import spatialmath.base as smb

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


class G1BaseController():

    def __init__(
        self,
        target_pose_topic = "rt/dolly_pose",
        robot_pose_topic = "rt/torso_pose",
        robot_state_topic = "rt/sportmodestate",
        base_command = "rt/cmd_vel",
    ):
        self.target_vel = [0.0,0.0,0.0] # [vx,vy,omega]
        self.command_publisher = ChannelPublisher(base_command, String_)
        self.command_publisher.Init()

        ## Subcribers
        self.target_pose_sub = ChannelSubscriber(target_pose_topic, PoseStamped_)
        self.target_pose_first_frame = False
        self.target_pose = Pose()
        self.target_pose_sub.Init(lambda x: self.pose_callback(x, self.target_pose_first_frame, self.target_pose), 1)

        self.robot_pose_sub = ChannelSubscriber(robot_pose_topic, PoseStamped_)
        self.robot_pose_first_frame = False
        self.robot_pose = Pose()
        self.robot_pose_sub.Init(lambda x:self.pose_callback(x, self.robot_pose_first_frame, self.robot_pose), 1)
        

        # self.robot_state_sub = ChannelSubscriber(robot_state_topic, SportModeState_)
        # self.robot_state_sub.Init(self.robot_state_callback, 1)
        self.robot_state_first_frame = False


        while (not self.robot_state_first_frame) or (not self.target_pose_first_frame) or (not self.robot_pose_first_frame):
            print("Waiting for topics")
            print(self.target_pose)
            print(self.robot_pose)
            time.sleep(1)

    def pose_callback(self, msg: PoseStamped_, first_frame_flag, pose: Pose):
        pos = msg.pose.position
        quat = msg.pose.orientation


        pose.x = pos.x
        pose.y = pos.y
        pose.z = pos.z
        pose.w = quat.w
        pose.rx = quat.x
        pose.ry = quat.y
        pose.rz = quat.z

        if not first_frame_flag:
            first_frame_flag = True

    # def callback_target_pose(self, msg: PoseStamped_):
    #     pos = msg.pose.position
    #     quat = msg.pose.orientation
    #
    #
    #
    #     if not self.target_pose_first_frame:
    #         self.target_pose_first_frame = True


if __name__ == "__main__":
    ChannelFactoryInitialize(id = 1, networkInterface="lo")

    base_controller = G1BaseController()
