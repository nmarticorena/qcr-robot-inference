"""Convert an OMPL SE(2) path into direct SONIC planner target frames.

SONIC's ``specific_target_positions`` input is a single four-frame token at
30 Hz. It is not a list of four distant navigation checkpoints. This module
therefore maintains progress along the full OMPL path and emits a rolling,
physically reachable four-frame root-pose horizon.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import time

import spatialmath as sm

from rs_imle_policy.configs.g1.locomotion import SonicWaypointConfig
from rs_imle_policy.controllers.g1.gear_sonic_interface import (
    LocomotionMode,
    PLANNER_WAYPOINT_COUNT,
    PlannerCommand,
    PlannerWaypoints,
)


class SonicWaypointController:
    """Build direct SONIC waypoint commands from a sampled OMPL path.

    Args:
        path: World-frame SE(2) poses returned by the OMPL planner.
        config: Direct-target tuning parameters.
        planner_origin: External world pose corresponding to SONIC's internal
            ``(x=0, y=0, yaw=0)`` initialization. Keep this fixed across OMPL
            replans within one SONIC session. Defaults to the first path pose.
        clock: Injectable monotonic clock for deterministic tests.
    """

    def __init__(
        self,
        path: Sequence[sm.SE2],
        config: SonicWaypointConfig | None = None,
        planner_origin: sm.SE2 | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        config = config or SonicWaypointConfig()
        if not path:
            raise ValueError("path must contain at least one pose")
        numeric_values = (
            config.speed,
            config.planner_frame_hz,
            config.target_lookahead,
            config.height,
            config.planner_root_height,
            config.goal_tolerance,
            config.goal_tolerance_theta,
            config.projection_search_distance,
            config.max_reference_lead,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("controller parameters must be finite")
        if config.speed <= 0.0:
            raise ValueError("speed must be positive")
        if config.planner_frame_hz <= 0.0:
            raise ValueError("planner_frame_hz must be positive")
        if config.target_lookahead < 0.0:
            raise ValueError("target_lookahead must be non-negative")
        if config.goal_tolerance < 0.0 or config.goal_tolerance_theta < 0.0:
            raise ValueError("goal tolerances must be non-negative")
        if config.projection_search_distance <= 0.0:
            raise ValueError("projection_search_distance must be positive")
        if config.max_reference_lead <= 0.0:
            raise ValueError("max_reference_lead must be positive")

        self.config = config
        self.path = tuple(path)
        self.speed = float(config.speed)
        self.planner_frame_hz = float(config.planner_frame_hz)
        self.target_lookahead = float(config.target_lookahead)
        self.locomotion_mode = LocomotionMode(config.locomotion_mode)
        self.height = float(config.height)
        # SONIC initializes its generated-motion world at x=y=yaw=0, rather
        # than at the external localization pose. Keep one fixed transform for
        # the whole SONIC session; do not reset it on every OMPL replan.
        self.planner_origin = planner_origin if planner_origin is not None else path[0]
        self.planner_root_height = float(config.planner_root_height)
        self.goal_tolerance = float(config.goal_tolerance)
        self.goal_tolerance_theta = float(config.goal_tolerance_theta)
        self.projection_search_distance = float(config.projection_search_distance)
        self.max_reference_lead = float(config.max_reference_lead)
        self._clock = clock

        cumulative_distance = [0.0]
        for start, end in zip(self.path, self.path[1:]):
            cumulative_distance.append(cumulative_distance[-1] + math.hypot(end.x - start.x, end.y - start.y))
        self._cumulative_distance = tuple(cumulative_distance)
        self._measured_progress = 0.0
        self._reference_progress = 0.0
        self._last_command_time: float | None = None

    @property
    def progress(self) -> float:
        """Time-progressing command reference along the path in metres."""

        return self._reference_progress

    @property
    def measured_progress(self) -> float:
        """Monotonic measured arc-length progress along the path in metres."""

        return self._measured_progress

    def reset(self) -> None:
        """Restart path progress from the first pose."""

        self._measured_progress = 0.0
        self._reference_progress = 0.0
        self._last_command_time = None

    def reached(self, pose: sm.SE2) -> bool:
        """Whether both final position and heading tolerances are satisfied."""

        goal = self.path[-1]
        position_error = math.hypot(goal.x - pose.x, goal.y - pose.y)
        heading_error = abs(math.remainder(goal.theta() - pose.theta(), 2.0 * math.pi))
        return position_error <= self.goal_tolerance and heading_error <= self.goal_tolerance_theta

    def command(self, pose: sm.SE2) -> PlannerCommand:
        """Return the next direct planner command for a measured robot pose."""
        if self.reached(pose):
            goal_yaw = (self.planner_origin.inv() * self.path[-1]).theta()
            return PlannerCommand(
                mode=LocomotionMode.IDLE,
                movement=(0.0, 0.0, 0.0),
                facing=(math.cos(goal_yaw), math.sin(goal_yaw), 0.0),
                speed=-1.0,
                height=self.height,
            )

        now = self._clock()
        if not math.isfinite(now):
            raise ValueError("clock must return a finite time")
        elapsed = 0.0 if self._last_command_time is None else min(0.5, max(0.0, now - self._last_command_time))
        self._last_command_time = now

        self._measured_progress = max(self._measured_progress, self._project(pose))
        # A target based solely on measured progress freezes when SONIC has
        # taken its first step but has not yet translated enough for the path
        # projection to move. Advance the command reference with wall time,
        # while bounding it relative to localization feedback.
        self._reference_progress = max(
            self._reference_progress,
            self._measured_progress,
        )
        self._reference_progress = min(
            self._cumulative_distance[-1],
            self._measured_progress + self.max_reference_lead,
            self._reference_progress + self.speed * elapsed,
        )
        frame_step = self.speed / self.planner_frame_hz
        target_start = self._reference_progress + self.speed * self.target_lookahead
        targets = [self._sample(target_start + frame_step * frame) for frame in range(PLANNER_WAYPOINT_COUNT)]
        planner_targets = [self.planner_origin.inv() * target for target in targets]
        planner_pose = self.planner_origin.inv() * pose
        headings = [target.theta() for target in planner_targets]
        positions = [(target.x, target.y, self.planner_root_height) for target in planner_targets]

        dx = planner_targets[-1].x - planner_pose.x
        dy = planner_targets[-1].y - planner_pose.y
        norm = math.hypot(dx, dy)
        movement = (0.0, 0.0, 0.0) if norm <= 1e-9 else (float(dx / norm), float(dy / norm), 0.0)
        facing_yaw = headings[-1]
        return PlannerCommand(
            mode=self.locomotion_mode,
            movement=movement,
            facing=(math.cos(facing_yaw), math.sin(facing_yaw), 0.0),
            speed=self.speed,
            height=self.height,
            waypoints=PlannerWaypoints.from_frames(positions, headings),
        )

    def _project(self, pose: sm.SE2) -> float:
        if len(self.path) == 1:
            return 0.0

        maximum_progress = min(
            self._cumulative_distance[-1],
            self._measured_progress + self.projection_search_distance,
        )
        best_distance_squared = math.inf
        best_progress = self._measured_progress
        for index, (start, end) in enumerate(zip(self.path, self.path[1:])):
            segment_start = self._cumulative_distance[index]
            segment_end = self._cumulative_distance[index + 1]
            if segment_end < self._measured_progress or segment_start > maximum_progress:
                continue

            dx = end.x - start.x
            dy = end.y - start.y
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-12:
                fraction = 0.0
            else:
                fraction = ((pose.x - start.x) * dx + (pose.y - start.y) * dy) / length_squared
                fraction = min(1.0, max(0.0, fraction))
            projected_x = start.x + fraction * dx
            projected_y = start.y + fraction * dy
            distance_squared = (pose.x - projected_x) ** 2 + (pose.y - projected_y) ** 2
            progress = segment_start + fraction * (segment_end - segment_start)
            if progress < self._measured_progress or progress > maximum_progress:
                continue
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_progress = progress
        return best_progress

    def _sample(self, progress: float) -> sm.SE2:
        progress = min(max(0.0, progress), self._cumulative_distance[-1])
        if len(self.path) == 1 or progress >= self._cumulative_distance[-1]:
            return self.path[-1]

        for index in range(len(self.path) - 1):
            start_distance = self._cumulative_distance[index]
            end_distance = self._cumulative_distance[index + 1]
            if progress > end_distance:
                continue
            segment_length = end_distance - start_distance
            fraction = 0.0 if segment_length <= 1e-12 else (progress - start_distance) / segment_length
            start = self.path[index]
            end = self.path[index + 1]
            yaw_delta = math.remainder(end.theta() - start.theta(), 2.0 * math.pi)
            return sm.SE2(
                start.x + fraction * (end.x - start.x),
                start.y + fraction * (end.y - start.y),
                start.theta() + fraction * yaw_delta,
            )
        return self.path[-1]


__all__ = ["SonicWaypointController"]
