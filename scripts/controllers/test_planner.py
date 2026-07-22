from matplotlib import pyplot as plt

from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from rs_imle_policy.configs.g1.locomotion import PathPlannerConfig
from rs_imle_policy.controllers.g1.locomotion.planning import DubinsPlanner
from rs_imle_policy.utils.dds import pose_stamped_to_se2


def main(domain_id: int = 1, network_interface: str = "lo"):
    ChannelFactoryInitialize(domain_id, network_interface)

    dolly_pose_sub = ChannelSubscriber("rt/dolly_pose", PoseStamped_)
    dolly_pose_sub.Init()
    robot_pose_sub = ChannelSubscriber("rt/torso_pose", PoseStamped_)
    robot_pose_sub.Init()

    goal_pose = pose_stamped_to_se2(dolly_pose_sub.Read(timeout=1))
    initial_pose = pose_stamped_to_se2(robot_pose_sub.Read(timeout=1))
    planner = DubinsPlanner(PathPlannerConfig(bounds=(-5, 5, -5, 5), turning_radius=0.2))
    planner.plan(initial_pose, goal_pose)
    planner.plot()
    plt.show()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
