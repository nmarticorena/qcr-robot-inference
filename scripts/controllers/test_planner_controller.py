"""Navigate the G1 to the dolly through SportMode or Gear-Sonic."""

from dataclasses import dataclass, field
import time

from loop_rate_limiters import RateLimiter
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from rs_imle_policy.configs.g1.locomotion import (
    DollyNavigationConfig,
    DollyVRConfig,
    WalkingBackend,
)
from rs_imle_policy.controllers.g1.locomotion.commands import (
    PlanarVelocity,
    VelocityInterface,
    create_velocity_interface,
)
from rs_imle_policy.controllers.g1.locomotion.dolly import DollyPlan, plan_to_dolly
from rs_imle_policy.controllers.g1.locomotion.gear_sonic import (
    GearSonicVelocityInterface,
)
from rs_imle_policy.controllers.g1.locomotion.pure_pursuit import (
    PurePursuitController,
)
from rs_imle_policy.controllers.g1.locomotion.vr_tracking import (
    CurrentHandPoseFK,
    dolly_center_velocity,
    interpolate_vr_poses,
    smoothstep,
    waist_relative_vr_targets,
)
from rs_imle_policy.utils.dds import pose_stamped_to_se2, reset_dolly, set_rope


@dataclass
class PlannerControllerConfig:
    navigation: DollyNavigationConfig = field(default_factory=DollyNavigationConfig)
    vr: DollyVRConfig = field(default_factory=DollyVRConfig)
    repeat: bool = True


def _read(subscriber: ChannelSubscriber):
    message = subscriber.Read(timeout=1)
    if message is None:
        raise TimeoutError("timed out waiting for a required DDS topic")
    return message


def _plot_plan(plan: DollyPlan, config: DollyNavigationConfig) -> None:
    ax = plan.planner.plot()
    obstacle = plan.obstacle
    corner = obstacle.pose * type(obstacle.pose)(-obstacle.length / 2.0, -obstacle.width / 2.0, 0.0)
    ax.add_patch(
        Rectangle(
            (corner.x, corner.y),
            obstacle.length,
            obstacle.width,
            angle=np.rad2deg(obstacle.pose.theta()),
            color="tab:gray",
            alpha=0.5,
            label="dolly",
        )
    )
    ax.add_patch(
        Circle(
            (plan.path[0].x, plan.path[0].y),
            config.dolly.robot_radius,
            fill=False,
            color="tab:orange",
            label="robot footprint",
        )
    )
    ax.legend()
    plt.show(block=False)
    plt.pause(0.1)


def _follow_path(
    plan: DollyPlan,
    pose_subscriber: ChannelSubscriber,
    command_interface: VelocityInterface,
    config: DollyNavigationConfig,
) -> None:
    controller = PurePursuitController(plan.path, config.pursuit)
    rate = RateLimiter(config.control_hz, warn=False)
    filtered = np.zeros(3, dtype=float)
    alpha = config.pursuit.velocity_filter_alpha
    if not 0.0 < alpha <= 1.0:
        raise ValueError("velocity_filter_alpha must be in (0, 1]")

    while True:
        pose = pose_stamped_to_se2(_read(pose_subscriber))
        raw = controller.command(pose)
        if controller.reached(pose):
            command_interface.send(PlanarVelocity())
            return
        filtered += alpha * (np.asarray(raw.as_tuple()) - filtered)
        if controller.debug and controller.debug.phase in {
            "initial alignment",
            "goal alignment",
        }:
            filtered[:2] = 0.0
        command_interface.send(PlanarVelocity(*filtered))
        if config.visualize:
            plt.pause(0.001)
        rate.sleep()


def _track_dolly_vr(
    interface: GearSonicVelocityInterface,
    dolly_subscriber: ChannelSubscriber,
    torso_subscriber: ChannelSubscriber,
    lowstate_subscriber: ChannelSubscriber,
    config: PlannerControllerConfig,
) -> None:
    hand_fk = CurrentHandPoseFK()
    start = hand_fk.vr_poses(_read(lowstate_subscriber))
    transition_start = time.monotonic()
    rate = RateLimiter(config.navigation.control_hz, warn=False)
    alpha = 0.0
    while alpha < 1.0:
        target = waist_relative_vr_targets(_read(dolly_subscriber), _read(torso_subscriber), config.vr)
        elapsed = time.monotonic() - transition_start
        alpha = 1.0 if config.vr.transition_duration == 0.0 else smoothstep(elapsed / config.vr.transition_duration)
        interface.set_vr_poses(*interpolate_vr_poses(start, target, alpha))
        rate.sleep()

    tracking_start = time.monotonic()
    try:
        while time.monotonic() - tracking_start < config.vr.tracking_duration:
            dolly_pose = _read(dolly_subscriber)
            torso_pose = _read(torso_subscriber)
            interface.set_vr_poses(*waist_relative_vr_targets(dolly_pose, torso_pose, config.vr))
            interface.send(PlanarVelocity(*dolly_center_velocity(dolly_pose, torso_pose, config.vr)))
            rate.sleep()
    finally:
        interface.stop()


def main(config: PlannerControllerConfig) -> None:
    navigation = config.navigation
    ChannelFactoryInitialize(
        navigation.dds.domain_id,
        navigation.dds.network_interface,
    )
    dolly_subscriber = ChannelSubscriber(navigation.dds.dolly_pose_topic, PoseStamped_)
    dolly_subscriber.Init()
    torso_subscriber = ChannelSubscriber(navigation.dds.torso_pose_topic, PoseStamped_)
    torso_subscriber.Init()
    lowstate_subscriber = ChannelSubscriber(navigation.dds.lowstate_topic, LowState_)
    lowstate_subscriber.Init()

    command_interface = create_velocity_interface(navigation.locomotion)
    gear_sonic = navigation.locomotion.backend is WalkingBackend.GEAR_SONIC
    rope_disabled = False
    try:
        command_interface.start()
        if gear_sonic:
            set_rope("disable", navigation.dds.elastic_band_topic)
            rope_disabled = True
        while True:
            reset_dolly()
            time.sleep(navigation.reset_delay)
            dolly_pose = pose_stamped_to_se2(_read(dolly_subscriber))
            robot_pose = pose_stamped_to_se2(_read(torso_subscriber))
            plan = plan_to_dolly(robot_pose, dolly_pose, navigation.dolly)
            if navigation.visualize:
                _plot_plan(plan, navigation)
            _follow_path(plan, torso_subscriber, command_interface, navigation)
            print("Goal reached")
            if gear_sonic and config.vr.enabled:
                assert isinstance(command_interface, GearSonicVelocityInterface)
                _track_dolly_vr(
                    command_interface,
                    dolly_subscriber,
                    torso_subscriber,
                    lowstate_subscriber,
                    config,
                )
            if not config.repeat:
                break
            input("Press Enter to plan again, or Ctrl+C to exit...")
            plt.close("all")
    except KeyboardInterrupt:
        print("\nStopping planner controller")
    finally:
        command_interface.close()
        if rope_disabled:
            set_rope("enable", navigation.dds.elastic_band_topic)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(PlannerControllerConfig))
