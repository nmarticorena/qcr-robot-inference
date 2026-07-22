"""Navigate to the dolly with Gear-Sonic direct planner targets."""

from dataclasses import dataclass, field, replace
import time

from loop_rate_limiters import RateLimiter
import spatialmath as sm
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

from rs_imle_policy.configs.g1.locomotion import (
    DDSConfig,
    DollyPlanningConfig,
    GearSonicConfig,
    SonicWaypointConfig,
)
from rs_imle_policy.controllers.g1.gear_sonic_waypoint_controller import (
    SonicWaypointController,
)
from rs_imle_policy.controllers.g1.locomotion.dolly import plan_to_dolly
from rs_imle_policy.controllers.g1.gear_sonic_interface import GearSonicInterface
from rs_imle_policy.utils.dds import pose_stamped_to_se2, reset_dolly, set_rope


@dataclass
class SonicDollyConfig:
    dds: DDSConfig = field(default_factory=DDSConfig)
    dolly: DollyPlanningConfig = field(default_factory=DollyPlanningConfig)
    gear_sonic: GearSonicConfig = field(default_factory=GearSonicConfig)
    waypoint: SonicWaypointConfig = field(default_factory=SonicWaypointConfig)
    command_hz: float = 10.0
    planner_replan_hz: float = 1.0
    reset_delay: float = 0.5
    repeat: bool = True


def _read(subscriber: ChannelSubscriber) -> PoseStamped_:
    message = subscriber.Read(timeout=1)
    if message is None:
        raise TimeoutError("timed out waiting for a required DDS pose")
    return message


def main(config: SonicDollyConfig) -> None:
    if not 0.0 < config.planner_replan_hz <= config.command_hz:
        raise ValueError("planner_replan_hz must be in (0, command_hz]")
    dds = config.dds
    ChannelFactoryInitialize(dds.domain_id, dds.network_interface)
    dolly_subscriber = ChannelSubscriber(dds.dolly_pose_topic, PoseStamped_)
    dolly_subscriber.Init()
    torso_subscriber = ChannelSubscriber(dds.torso_pose_topic, PoseStamped_)
    torso_subscriber.Init()

    sonic_config = replace(
        config.gear_sonic,
        publish_hz=max(config.gear_sonic.publish_hz, config.command_hz),
    )
    sonic = GearSonicInterface(sonic_config)
    rate = RateLimiter(config.command_hz, warn=False)
    planner_origin: sm.SE2 | None = None
    rope_disabled = False
    try:
        sonic.start()
        set_rope("disable", dds.elastic_band_topic)
        rope_disabled = True
        while True:
            reset_dolly()
            time.sleep(config.reset_delay)
            dolly_pose = pose_stamped_to_se2(_read(dolly_subscriber))
            robot_pose = pose_stamped_to_se2(_read(torso_subscriber))
            if planner_origin is None:
                planner_origin = robot_pose
            plan = plan_to_dolly(robot_pose, dolly_pose, config.dolly)
            controller = SonicWaypointController(
                plan.path,
                config.waypoint,
                planner_origin,
            )
            update_period = 1.0 / config.planner_replan_hz
            last_update = -float("inf")
            while True:
                pose = pose_stamped_to_se2(_read(torso_subscriber))
                command = controller.command(pose)
                now = time.monotonic()
                if controller.reached(pose) or now - last_update >= update_period:
                    sonic.set_planner_command(command)
                    last_update = now
                if controller.reached(pose):
                    sonic.clear_waypoints(facing_yaw=(planner_origin.inv() * plan.goal).theta())
                    break
                rate.sleep()
            print("Goal reached")
            if not config.repeat:
                break
            input("Press Enter to plan again, or Ctrl+C to exit...")
    except KeyboardInterrupt:
        print("\nStopping direct SONIC waypoint controller")
    finally:
        sonic.close()
        if rope_disabled:
            set_rope("enable", dds.elastic_band_topic)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(SonicDollyConfig))
