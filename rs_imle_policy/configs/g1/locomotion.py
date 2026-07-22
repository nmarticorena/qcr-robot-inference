"""Configuration for G1 planar locomotion and dolly navigation."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class WalkingBackend(StrEnum):
    """Available G1 walking command transports."""

    SPORT_MODE = "sportsmode"
    GEAR_SONIC = "gear-sonic"


@dataclass
class DDSConfig:
    domain_id: int = 1
    network_interface: str = "lo"
    torso_pose_topic: str = "rt/torso_pose"
    dolly_pose_topic: str = "rt/dolly_pose"
    lowstate_topic: str = "rt/lowstate"
    sport_state_topic: str = "rt/sportmodestate"
    elastic_band_topic: str = "rt/elastic_band"


@dataclass
class VelocityLimits:
    max_vx: float = 0.8
    max_vy: float = 0.5
    max_omega: float = 1.0
    max_ax: float = 0.3
    max_ay: float = 0.2
    max_alpha: float = 0.8


@dataclass
class PIDGainsConfig:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.0
    output_limit: float = 1.0


@dataclass
class TargetStepConfig:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    delay: float = 2.0


@dataclass
class PathPlannerConfig:
    bounds: tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0)
    turning_radius: float = 0.2
    timeout: float = 1.0
    samples: int = 200
    validity_resolution: float = 0.01
    algorithm: Literal["rrt-connect", "rrt-star"] = "rrt-star"


@dataclass
class PurePursuitConfig:
    speed: float = 0.3
    lookahead: float = 0.5
    goal_tolerance: float = 0.1
    goal_tolerance_theta: float = 0.1
    alignment_tolerance: float = 0.1
    alignment_kp: float = 1.5
    alignment_speed: float = 0.1
    velocity_filter_alpha: float = 0.2
    limits: VelocityLimits = field(default_factory=VelocityLimits)


@dataclass
class SportModeConfig:
    command_topic: str = "rt/run_command/cmd"
    command_height: float = 0.8


@dataclass
class GearSonicConfig:
    endpoint: str = "tcp://*:5556"
    publish_hz: float = 20.0
    locomotion_mode: int | None = None
    height: float = -1.0
    connection_timeout: float = 10.0
    connection_delay: float = 0.5
    command_repetitions: int = 3


@dataclass
class SonicWaypointConfig:
    speed: float = 0.3
    planner_frame_hz: float = 30.0
    target_lookahead: float = 1.0
    max_reference_lead: float = 0.5
    locomotion_mode: int = 1
    height: float = -1.0
    planner_root_height: float = 0.8
    goal_tolerance: float = 0.1
    goal_tolerance_theta: float = 0.1
    projection_search_distance: float = 1.0


@dataclass
class LocomotionConfig:
    backend: WalkingBackend = WalkingBackend.SPORT_MODE
    sport_mode: SportModeConfig = field(default_factory=SportModeConfig)
    gear_sonic: GearSonicConfig = field(default_factory=GearSonicConfig)

    def __post_init__(self) -> None:
        self.backend = WalkingBackend(self.backend)


@dataclass
class BasePIDControllerConfig:
    dds: DDSConfig = field(default_factory=DDSConfig)
    locomotion: LocomotionConfig = field(default_factory=LocomotionConfig)
    limits: VelocityLimits = field(default_factory=VelocityLimits)
    hz: float = 50.0
    x_gains: PIDGainsConfig = field(default_factory=lambda: PIDGainsConfig(6.0, 6.6666667, 3.564, 0.12, 0.8))
    y_gains: PIDGainsConfig = field(default_factory=lambda: PIDGainsConfig(4.0, 7.2727273, 1.452, 0.06875, 0.5))
    yaw_gains: PIDGainsConfig = field(default_factory=lambda: PIDGainsConfig(16.0, 40.0, 4.224, 0.025, 1.0))
    enable_x: bool = True
    enable_y: bool = True
    enable_yaw: bool = True
    log: bool = False
    target_step: TargetStepConfig | None = None


@dataclass
class DollyPlanningConfig:
    planner: PathPlannerConfig = field(default_factory=PathPlannerConfig)
    goal_offset: float = -0.9
    robot_radius: float = 0.3
    cart_box_length: float = 1.6
    cart_box_width: float = 1.0
    cart_box_gap: float = 0.0


@dataclass
class DollyNavigationConfig:
    dds: DDSConfig = field(default_factory=DDSConfig)
    locomotion: LocomotionConfig = field(default_factory=LocomotionConfig)
    dolly: DollyPlanningConfig = field(default_factory=DollyPlanningConfig)
    pursuit: PurePursuitConfig = field(default_factory=PurePursuitConfig)
    control_hz: float = 50.0
    reset_delay: float = 0.5
    visualize: bool = True


@dataclass
class DollyVRConfig:
    enabled: bool = True
    transition_duration: float = 3.0
    tracking_speed: float = 0.8
    tracking_duration: float = 20.0
    yaw_kp: float = 1.5
    max_omega: float = 0.5
    left_handle: tuple[float, float, float] = (-0.45, 0.13, 0.35)
    right_handle: tuple[float, float, float] = (-0.45, -0.13, 0.35)


__all__ = [
    "DDSConfig",
    "BasePIDControllerConfig",
    "DollyNavigationConfig",
    "DollyPlanningConfig",
    "DollyVRConfig",
    "GearSonicConfig",
    "LocomotionConfig",
    "PathPlannerConfig",
    "PIDGainsConfig",
    "PurePursuitConfig",
    "SonicWaypointConfig",
    "SportModeConfig",
    "TargetStepConfig",
    "VelocityLimits",
    "WalkingBackend",
]
