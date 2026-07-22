"""Send a manually specified torso path to Gear-Sonic."""

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import time

from loop_rate_limiters import RateLimiter
import spatialmath.base as smb
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

from rs_imle_policy.configs.g1.locomotion import (
    DDSConfig,
    GearSonicConfig,
    SonicWaypointConfig,
)
from rs_imle_policy.controllers.g1.gear_sonic_waypoint_controller import (
    SonicWaypointController,
)
from rs_imle_policy.controllers.g1.gear_sonic_interface import GearSonicInterface
from rs_imle_policy.controllers.g1.locomotion.path_io import (
    PlanFrame,
    YawUnit,
    load_plan,
    make_world_path,
)
from rs_imle_policy.utils.dds import pose_stamped_to_se2, set_rope


@dataclass
class ManualTorsoPlanConfig:
    plan: Path
    dds: DDSConfig = field(default_factory=DDSConfig)
    gear_sonic: GearSonicConfig = field(default_factory=GearSonicConfig)
    waypoint: SonicWaypointConfig = field(default_factory=SonicWaypointConfig)
    frame: PlanFrame | None = None
    yaw_unit: YawUnit | None = None
    command_hz: float = 10.0
    planner_replan_hz: float = 1.0
    stall_timeout: float = 10.0
    confirm: bool = True
    restore_rope: bool = True


def _read(subscriber: ChannelSubscriber) -> PoseStamped_:
    message = subscriber.Read(timeout=1)
    if message is None:
        raise TimeoutError("timed out waiting for torso pose")
    return message


def main(config: ManualTorsoPlanConfig) -> None:
    if not 0.0 < config.planner_replan_hz <= config.command_hz:
        raise ValueError("planner_replan_hz must be in (0, command_hz]")
    if config.stall_timeout < 0.0:
        raise ValueError("stall_timeout must be non-negative")
    manual_waypoints, frame = load_plan(config.plan, config.frame, config.yaw_unit)
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.network_interface)
    torso_subscriber = ChannelSubscriber(config.dds.torso_pose_topic, PoseStamped_)
    torso_subscriber.Init()
    start = pose_stamped_to_se2(_read(torso_subscriber))
    path = make_world_path(manual_waypoints, frame, start)

    print(f"Manual plan ({frame} input -> world frame):")
    for index, pose in enumerate(path):
        print(f"  {index:02d}: x={pose.x:+.3f}, y={pose.y:+.3f}, yaw={math.degrees(pose.theta()):+.1f} deg")
    if config.confirm:
        answer = input("Send this torso plan to Gear-Sonic? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            return

    controller = SonicWaypointController(
        path,
        config.waypoint,
        start,
    )
    sonic = GearSonicInterface(
        replace(
            config.gear_sonic,
            publish_hz=max(config.gear_sonic.publish_hz, config.command_hz),
        )
    )
    rate = RateLimiter(config.command_hz, warn=False)
    update_period = 1.0 / config.planner_replan_hz
    last_update = -math.inf
    last_motion_time = time.monotonic()
    last_motion_pose = start
    rope_disabled = False
    try:
        sonic.start()
        set_rope("disable", config.dds.elastic_band_topic)
        rope_disabled = True
        while True:
            pose = pose_stamped_to_se2(_read(torso_subscriber))
            command = controller.command(pose)
            now = time.monotonic()
            translation = math.hypot(pose.x - last_motion_pose.x, pose.y - last_motion_pose.y)
            rotation = abs(float(smb.angdiff(pose.theta(), last_motion_pose.theta())))
            if translation >= 0.01 or rotation >= math.radians(2.0):
                last_motion_pose = pose
                last_motion_time = now
            if controller.reached(pose) or now - last_update >= update_period:
                sonic.set_planner_command(command)
                last_update = now
            if controller.reached(pose):
                sonic.clear_waypoints(facing_yaw=(start.inv() * path[-1]).theta())
                print("Torso plan reached")
                return
            if config.stall_timeout and now - last_motion_time >= config.stall_timeout:
                sonic.clear_waypoints(facing_yaw=(start.inv() * pose).theta())
                raise RuntimeError("torso pose did not change while SONIC was commanded")
            rate.sleep()
    except KeyboardInterrupt:
        print("\nStopping manual torso plan")
    finally:
        sonic.close()
        if rope_disabled and config.restore_rope:
            set_rope("enable", config.dds.elastic_band_topic)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(ManualTorsoPlanConfig))
