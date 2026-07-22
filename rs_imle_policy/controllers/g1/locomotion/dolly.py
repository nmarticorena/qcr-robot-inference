"""Dolly-specific geometry and path construction."""

from dataclasses import dataclass

import spatialmath as sm

from rs_imle_policy.configs.g1.locomotion import DollyPlanningConfig
from rs_imle_policy.controllers.g1.locomotion.planning import (
    BoxObstacle,
    CylinderRobotCollisionChecker,
    DubinsPlanner,
)


@dataclass(frozen=True)
class DollyPlan:
    goal: sm.SE2
    obstacle: BoxObstacle
    path: tuple[sm.SE2, ...]
    planner: DubinsPlanner


def plan_to_dolly(
    robot_pose: sm.SE2,
    dolly_pose: sm.SE2,
    config: DollyPlanningConfig,
) -> DollyPlan:
    goal = dolly_pose * sm.SE2(config.goal_offset, 0.0, 0.0)
    obstacle_pose = dolly_pose * sm.SE2(
        config.cart_box_gap + config.cart_box_length / 2.0,
        0.0,
        0.0,
    )
    obstacle = BoxObstacle(
        pose=obstacle_pose,
        length=config.cart_box_length,
        width=config.cart_box_width,
    )
    checker = CylinderRobotCollisionChecker(config.robot_radius, [obstacle])
    planner = DubinsPlanner(config.planner, is_valid=checker)
    path = tuple(planner.plan(robot_pose, goal))
    return DollyPlan(goal, obstacle, path, planner)


__all__ = ["DollyPlan", "plan_to_dolly"]
