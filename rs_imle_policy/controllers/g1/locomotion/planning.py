"""Planar path planning and footprint collision checking."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from spatialmath import SE2

from rs_imle_policy.configs.g1.locomotion import PathPlannerConfig


def wrap_to_pi(yaw: float) -> float:
    return math.remainder(float(yaw), 2.0 * math.pi)


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, space_information, is_valid: Callable[[SE2], bool]):
        super().__init__(space_information)
        self._is_valid = is_valid

    def isValid(self, state) -> bool:  # noqa: N802 - OMPL callback name
        return self._is_valid(SE2(state.getX(), state.getY(), state.getYaw()))


@dataclass(frozen=True)
class BoxObstacle:
    pose: SE2
    length: float
    width: float

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("obstacle dimensions must be positive")

    def collides_with_circle(self, pose: SE2, radius: float) -> bool:
        if radius < 0.0:
            raise ValueError("robot radius must be non-negative")
        dx = pose.x - self.pose.x
        dy = pose.y - self.pose.y
        yaw = self.pose.theta()
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        closest_x = np.clip(local_x, -self.length / 2.0, self.length / 2.0)
        closest_y = np.clip(local_y, -self.width / 2.0, self.width / 2.0)
        return bool((local_x - closest_x) ** 2 + (local_y - closest_y) ** 2 <= radius**2)


class CylinderRobotCollisionChecker:
    def __init__(self, radius: float, obstacles: Sequence[BoxObstacle]):
        if radius < 0.0:
            raise ValueError("robot radius must be non-negative")
        self.radius = float(radius)
        self.obstacles = tuple(obstacles)

    def __call__(self, pose: SE2) -> bool:
        return not any(obstacle.collides_with_circle(pose, self.radius) for obstacle in self.obstacles)


class DubinsPlanner:
    """Plan forward-only, curvature-limited paths between SE(2) poses."""

    def __init__(
        self,
        config: PathPlannerConfig | None = None,
        is_valid: Callable[[SE2], bool] | None = None,
    ) -> None:
        self.config = config or PathPlannerConfig()
        xmin, xmax, ymin, ymax = self.config.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("planner bounds must have increasing limits")
        if self.config.turning_radius <= 0.0:
            raise ValueError("turning_radius must be positive")
        if self.config.samples < 2:
            raise ValueError("samples must be at least two")
        if not 0.0 < self.config.validity_resolution <= 1.0:
            raise ValueError("validity_resolution must be in (0, 1]")

        self.space = ob.DubinsStateSpace(self.config.turning_radius, False)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, xmin)
        bounds.setHigh(0, xmax)
        bounds.setLow(1, ymin)
        bounds.setHigh(1, ymax)
        self.space.setBounds(bounds)
        self.is_valid = is_valid or (lambda _pose: True)
        self.path: list[SE2] | None = None

    def _state(self, pose: SE2):
        state = self.space.allocState()
        state.setX(pose.x)
        state.setY(pose.y)
        state.setYaw(wrap_to_pi(pose.theta()))
        return state

    def plan(self, start: SE2, goal: SE2) -> list[SE2]:
        setup = og.SimpleSetup(self.space)
        space_information = setup.getSpaceInformation()
        setup.setStateValidityChecker(ValidityChecker(space_information, self.is_valid))
        space_information.setStateValidityCheckingResolution(self.config.validity_resolution)
        setup.setStartAndGoalStates(self._state(start), self._state(goal))
        planner_type = {
            "rrt-connect": og.RRTConnect,
            "rrt-star": og.RRTstar,
        }[self.config.algorithm]
        setup.setPlanner(planner_type(space_information))
        setup.setOptimizationObjective(ob.PathLengthOptimizationObjective(space_information))
        if not setup.solve(self.config.timeout):
            raise RuntimeError("no collision-free Dubins path found")
        path = setup.getSolutionPath()
        path.interpolate(self.config.samples)
        self.path = [SE2(state.getX(), state.getY(), state.getYaw()) for state in path.getStates()]
        return self.path

    def plot(self, ax: plt.Axes | None = None, arrow_every: int = 20) -> plt.Axes:
        if self.path is None:
            raise RuntimeError("call plan() before plot()")
        ax = ax or plt.subplots(figsize=(7, 7))[1]
        xy = np.asarray([(pose.x, pose.y) for pose in self.path])
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", label="Dubins path")
        for pose in self.path[::arrow_every]:
            ax.arrow(
                pose.x,
                pose.y,
                0.18 * np.cos(pose.theta()),
                0.18 * np.sin(pose.theta()),
                head_width=0.07,
                color="black",
                length_includes_head=True,
            )
        ax.scatter(*xy[0], color="green", s=60, label="start", zorder=3)
        ax.scatter(*xy[-1], color="black", s=60, label="goal", zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.grid()
        return ax


__all__ = [
    "BoxObstacle",
    "CylinderRobotCollisionChecker",
    "DubinsPlanner",
    "ValidityChecker",
    "wrap_to_pi",
]
