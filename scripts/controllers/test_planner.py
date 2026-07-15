from matplotlib import pyplot as plt
import spatialmath as sm
import spatialmath.base as smb

from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from rs_imle_policy.controllers.g1.base_planner import ReedsSheppPlanner



def main(domain_id: int = 1, networkInterface = "lo"):
    ChannelFactoryInitialize(domain_id, networkInterface)

    dolly_pose_sub = ChannelSubscriber("rt/dolly_pose", PoseStamped_)
    dolly_pose_sub.Init()
    robot_pose_sub = ChannelSubscriber("rt/torso_pose", PoseStamped_)
    robot_pose_sub.Init()

    x_wd = dolly_pose_sub.Read(timeout=1)
    x_wr = robot_pose_sub.Read(timeout=1)

    planner = ReedsSheppPlanner(bounds=(-5, 5, -5, 5), turning_radius=0.2)
    goal_yaw = sm.UnitQuaternion(x_wd.pose.orientation.w,(x_wd.pose.orientation.x, x_wd.pose.orientation.y, x_wd.pose.orientation.z)).rpy()[2]
    goal_pose = sm.SE2(x_wd.pose.position.x, x_wd.pose.position.y, goal_yaw)
    initial_yaw = sm.UnitQuaternion(x_wr.pose.orientation.w, (x_wr.pose.orientation.x, x_wr.pose.orientation.y, x_wr.pose.orientation.z)).rpy()[2]
    initial_pose = sm.SE2(x_wr.pose.position.x, x_wr.pose.position.y, initial_yaw) 

    path = planner.plan(initial_pose, goal_pose, timeout=1.0, samples=200)
    planner.plot()
    plt.show()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
