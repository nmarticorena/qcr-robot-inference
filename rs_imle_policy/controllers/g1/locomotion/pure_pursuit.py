"""Pure-pursuit path tracking independent of command transport."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import spatialmath as sm
import spatialmath.base as smb

from rs_imle_policy.configs.g1.locomotion import PurePursuitConfig
from rs_imle_policy.controllers.g1.locomotion.commands import PlanarVelocity


@dataclass(frozen=True)
class PurePursuitDebug:
    phase: str
    target_index: int
    target_x: float
    target_y: float
    desired_heading: float
    heading_error: float
    distance: float
    local_x: float
    local_y: float
    curvature: float
    velocity: PlanarVelocity

    @property
    def vx(self) -> float:
        return self.velocity.vx

    @property
    def omega(self) -> float:
        return self.velocity.omega


class PurePursuitController:
    def __init__(
        self,
        path: Sequence[sm.SE2],
        config: PurePursuitConfig | None = None,
    ) -> None:
        if not path:
            raise ValueError("path must contain at least one pose")
        self.path = tuple(path)
        self.config = config or PurePursuitConfig()
        if self.config.lookahead <= 0.0 or self.config.speed <= 0.0:
            raise ValueError("lookahead and speed must be positive")
        if self.config.alignment_kp <= 0.0:
            raise ValueError("alignment_kp must be positive")
        if self.config.alignment_tolerance < 0.0:
            raise ValueError("alignment_tolerance must be non-negative")
        self.target_index = 0
        self.aligned = False
        self.debug: PurePursuitDebug | None = None

    def reached(self, pose: sm.SE2) -> bool:
        goal = self.path[-1]
        return (
            float(np.hypot(goal.x - pose.x, goal.y - pose.y)) <= self.config.goal_tolerance
            and abs(float(smb.angdiff(goal.theta(), pose.theta()))) <= self.config.goal_tolerance_theta
        )

    def _turn(self, error: float) -> float:
        return float(
            np.clip(
                self.config.alignment_kp * error,
                -self.config.limits.max_omega,
                self.config.limits.max_omega,
            )
        )

    def _result(
        self,
        phase: str,
        target: sm.SE2,
        desired_heading: float,
        heading_error: float,
        distance: float,
        local_x: float,
        local_y: float,
        curvature: float,
        velocity: PlanarVelocity,
    ) -> PlanarVelocity:
        self.debug = PurePursuitDebug(
            phase,
            self.target_index,
            float(target.x),
            float(target.y),
            float(desired_heading),
            float(heading_error),
            float(distance),
            float(local_x),
            float(local_y),
            float(curvature),
            velocity,
        )
        return velocity

    def command(self, pose: sm.SE2) -> PlanarVelocity:
        goal = self.path[-1]
        if self.reached(pose):
            return self._result(
                "reached",
                goal,
                goal.theta(),
                0.0,
                float(np.hypot(goal.x - pose.x, goal.y - pose.y)),
                0.0,
                0.0,
                0.0,
                PlanarVelocity(),
            )

        while self.target_index < len(self.path) - 1:
            target = self.path[self.target_index]
            if np.hypot(target.x - pose.x, target.y - pose.y) >= self.config.lookahead:
                break
            self.target_index += 1

        target = self.path[self.target_index]
        dx, dy = target.x - pose.x, target.y - pose.y
        yaw = pose.theta()
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        distance_squared = dx * dx + dy * dy
        distance = float(np.sqrt(distance_squared))
        curvature = 0.0 if distance_squared <= 1e-12 else 2.0 * local_y / distance_squared
        desired_heading = float(np.arctan2(dy, dx))
        heading_error = float(smb.angdiff(desired_heading, yaw))

        goal_distance = float(np.hypot(goal.x - pose.x, goal.y - pose.y))
        if goal_distance <= self.config.goal_tolerance:
            goal_error = float(smb.angdiff(goal.theta(), yaw))
            return self._result(
                "goal alignment",
                goal,
                goal.theta(),
                goal_error,
                goal_distance,
                local_x,
                local_y,
                curvature,
                PlanarVelocity(omega=self._turn(goal_error)),
            )

        if not self.aligned:
            if abs(heading_error) > self.config.alignment_tolerance:
                velocity = PlanarVelocity(
                    vx=min(self.config.alignment_speed, self.config.limits.max_vx),
                    omega=self._turn(heading_error),
                )
                phase = "initial arc alignment" if velocity.vx else "initial alignment"
                return self._result(
                    phase,
                    target,
                    desired_heading,
                    heading_error,
                    distance,
                    local_x,
                    local_y,
                    curvature,
                    velocity,
                )
            self.aligned = True

        speed = min(self.config.speed, self.config.limits.max_vx)
        if curvature:
            speed = min(speed, self.config.limits.max_omega / abs(curvature))
        vx = -speed if local_x < 0.0 else speed
        return self._result(
            "pure pursuit",
            target,
            desired_heading,
            heading_error,
            distance,
            local_x,
            local_y,
            curvature,
            PlanarVelocity(vx=vx, omega=curvature * vx),
        )


__all__ = ["PurePursuitController", "PurePursuitDebug"]
