from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from ompl import base as ob
from ompl import geometric as og

def wrap_to_pi(yaw: float) -> float:
    return (yaw + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # radians

class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, space_information, is_valid):
        super().__init__(space_information)
        self.is_valid = is_valid

    def isValid(self, state):
        pose = Pose2D(state.getX(), state.getY(), state.getYaw())
        return self.is_valid(pose)


class ReedsSheppPlanner:
    """Plan forward/reverse, curvature-limited paths between SE(2) poses."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
        turning_radius: float = 0.5,
        is_valid: Callable[[Pose2D], bool] | None = None,
        planner: type[og.RRTConnect| og.RRTstar] = og.RRTstar,
    ):
        xmin, xmax, ymin, ymax = bounds
        self.space = ob.DubinsStateSpace(turning_radius, True)
        xy_bounds = ob.RealVectorBounds(2)
        xy_bounds.setLow(0, xmin)
        xy_bounds.setHigh(0, xmax)
        xy_bounds.setLow(1, ymin)
        xy_bounds.setHigh(1, ymax)
        self.space.setBounds(xy_bounds)
        print("bounds:", self.space.getBounds().low, self.space.getBounds().high)
        self.is_valid = is_valid or (lambda pose: True)
        self.path: list[Pose2D] | None = None
        self.planner = planner

    def _state(self, pose: Pose2D):
        state = self.space.allocState()
        state.setX(pose.x)
        state.setY(pose.y)
        state.setYaw(wrap_to_pi(pose.yaw))
        return state

    

    def plan(
        self, start: Pose2D, goal: Pose2D, timeout: float = 1.0, samples: int = 200
    ) -> list[Pose2D]:
        """Return equally spaced poses, or raise RuntimeError if no path is found."""

        setup = og.SimpleSetup(self.space)

        start_state = self._state(start)
        goal_state = self._state(goal)

        print("start bounds:", self.space.satisfiesBounds(start_state))
        print("goal bounds: ", self.space.satisfiesBounds(goal_state))

        setup.setStartAndGoalStates(start_state, goal_state)

        checker = ValidityChecker(setup.getSpaceInformation(), self.is_valid)
        setup.setStateValidityChecker(checker)

        setup.setStartAndGoalStates(self._state(start), self._state(goal))
        setup.setPlanner(self.planner(setup.getSpaceInformation()))
        setup.setOptimizationObjective(
           ob.PathLengthOptimizationObjective(setup.getSpaceInformation())
        )

        if not setup.solve(timeout):
            raise RuntimeError("No collision-free Reeds-Shepp path found")

        path = setup.getSolutionPath()
        path.interpolate(samples)
        self.path = [Pose2D(s.getX(), s.getY(), s.getYaw()) for s in path.getStates()]
        return self.path

    def plot(self, ax: plt.Axes | None = None, arrow_every: int = 20) -> plt.Axes:
        if self.path is None:
            raise RuntimeError("Call plan() before plot().")

        ax = ax or plt.subplots(figsize=(7, 7))[1]
        poses = self.path
        xy = np.array([(p.x, p.y) for p in poses])

        # A local displacement expressed opposite to the robot's heading means reverse motion.
        heading = np.array([[np.cos(p.yaw), np.sin(p.yaw)] for p in poses[:-1]])
        displacement = np.diff(xy, axis=0)
        reverse = np.einsum("ij,ij->i", displacement, heading) < 0.0
        for i in range(len(displacement)):
            ax.plot(xy[i : i + 2, 0], xy[i : i + 2, 1], color="tab:red" if reverse[i] else "tab:blue")

        for i in range(0, len(poses), arrow_every):
            p = poses[i]
            ax.arrow(p.x, p.y, 0.18 * np.cos(p.yaw), 0.18 * np.sin(p.yaw),
                     head_width=0.07, color="black", length_includes_head=True)

        ax.scatter(*xy[0], color="green", s=60, label="start", zorder=3)
        ax.scatter(*xy[-1], color="black", s=60, label="goal", zorder=3)
        ax.plot([], [], color="tab:blue", label="forward")
        ax.plot([], [], color="tab:red", label="reverse")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.grid()
        return ax


if __name__ == "__main__":
    planner = ReedsSheppPlanner(bounds=(-4, 4, -4, 4), turning_radius=0.1, planner= og.RRTstar)
    rrt_connectplanner = ReedsSheppPlanner(bounds=(-4, 4, -4, 4), turning_radius=0.1, planner=og.RRTConnect)
    start = Pose2D(-2.0, -1.5, np.deg2rad(np.random.uniform(-180, 180)))
    goal = Pose2D(-1, -1.0, np.deg2rad(np.random.uniform(-180, 180)))

    trajectory = planner.plan(start, goal)
    trajectory_2 = rrt_connectplanner.plan(start, goal)
    print(f"Generated {len(trajectory)} poses")
    planner.plot()
    rrt_connectplanner.plot()
    plt.show()
