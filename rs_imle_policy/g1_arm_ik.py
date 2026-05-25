"""
IK solver for the G1 Arms, it relies on Pink, and pinnochio
the collision geometry where approximated by convex hulls
"""

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pink
import pinocchio as pin
import viser
from pink import solve_ik
from pink.barriers import PositionBarrier, SelfCollisionBarrier
from pink.limits import AccelerationLimit
from pink.tasks import DampingTask, FrameTask, PostureTask
from pink.utils import process_collision_pairs
from pink.visualization import start_viser_visualizer
from scipy.spatial.transform import Rotation

from rs_imle_policy.utils.collision_utils import force_convex_collision_geometry
from rs_imle_policy.utils.urdf_utils import (
    DEFAULT_LOCKED_JOINTS,
    INSPIRE_DFQ_HAND_LINKS,
    LEG_LINKS,
    INSPIRE_FTP_HAND_LINKS,
)
from rs_imle_policy.configs.g1_configs import G1IKConfig, PositionBarrierBounds


@dataclass
class Targets:
    left: pin.SE3
    right: pin.SE3


class G1ReducedPinkIK:
    def __init__(
        self,
        config: G1IKConfig = G1IKConfig(),
        locked_joints: Iterable[str] = DEFAULT_LOCKED_JOINTS,
        visualize: bool = False,
        spawn_visualizer: bool = True,
        q0: np.ndarray | None = None,
        left_eef_position_bounds: PositionBarrierBounds | None = None,
        right_eef_position_bounds: PositionBarrierBounds | None = None,
    ):
        self.config = config
        self.visualize = visualize
        self.solver = config.solver

        self._setup_robot(locked_joints)
        self._setup_ee_frames(config.ee_offset)

        self.left_eef_position_bounds = left_eef_position_bounds or self._default_position_bounds()
        self.right_eef_position_bounds = right_eef_position_bounds or self._default_position_bounds()
        if q0 is None:
            q0 = pin.neutral(self.robot.model)

        self.configuration = pink.Configuration(
            self.robot.model,
            self.robot.data,
            q0,
            collision_model=self.robot.collision_model,
            collision_data=self.robot.collision_data,
        )

        self._setup_tasks()
        self._setup_barriers()
        self._setup_limits()

        if visualize:
            self._setup_visualizer(spawn_visualizer)

    def _setup_robot(self, locked_joints: Iterable[str]):
        srdf_path = self.config.srdf_path
        print("Loading robot model")
        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.config.urdf_path,
            ["assets/"],
            None,  # Not to use a freeflyer base
        )
        self.robot.collision_model.addAllCollisionPairs()
        converted = force_convex_collision_geometry(self.robot, inflation=self.config.inflation)
        print(f"Converted {converted} collision geometries to convex hulls")
        self._remove_collision_geometry(LEG_LINKS)
        self._remove_collision_geometry(INSPIRE_FTP_HAND_LINKS)
        self._remove_collision_geometry(INSPIRE_DFQ_HAND_LINKS)
        self.robot.collision_model.addAllCollisionPairs()  # rebuild pairs without legs

        if srdf_path and os.path.exists(srdf_path):
            pin.removeCollisionPairs(self.robot.model, self.robot.collision_model, srdf_path)
            self.robot.collision_data = process_collision_pairs(
                self.robot.model,
                self.robot.collision_model,
                srdf_path,
            )
        else:
            self.robot.collision_data = pin.GeometryData(self.robot.collision_model)

        # Remove locked joints
        joint_ids_to_lock = set([self.robot.model.getJointId(name) for name in locked_joints])
        reference_configuration = pin.neutral(self.robot.model)
        self.robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joint_ids_to_lock,
            reference_configuration=reference_configuration,
        )

        if not hasattr(self.robot, "collision_data") or self.robot.collision_data is None:
            self.robot.collision_data = pin.GeometryData(self.robot.collision_model)

    def _remove_collision_geometry(self, links_to_remove: Iterable[str]) -> None:
        """Remove all geometry objects associated with given link names."""
        names_to_remove = [
            g.name
            for g in self.robot.collision_model.geometryObjects
            if any(link in g.name for link in links_to_remove)
        ]
        for name in names_to_remove:
            self.robot.collision_model.removeGeometryObject(name)
        print(f"Removed {len(names_to_remove)} leg collision geometries")

    def _setup_ee_frames(self, ee_offset: float):
        self._add_ee_frame("L_ee", "left_wrist_yaw_joint", ee_offset)
        self._add_ee_frame("R_ee", "right_wrist_yaw_joint", ee_offset)
        self.robot.data = self.robot.model.createData()  # synch pin model
        self.left_ee_id = self.robot.model.getFrameId("L_ee")
        self.right_ee_id = self.robot.model.getFrameId("R_ee")

    def _setup_tasks(self):
        self.tasks = []
        if self.config.pos_cost > 0 or self.config.ori_cost > 0:
            print("Include position tasks")
            self.left_task = FrameTask(
                "L_ee",
                position_cost=self.config.pos_cost,
                orientation_cost=self.config.ori_cost,
            )
            self.right_task = FrameTask(
                "R_ee",
                position_cost=self.config.pos_cost,
                orientation_cost=self.config.ori_cost,
            )
            self.tasks.extend([self.left_task, self.right_task])
        if self.config.posture_cost > 0:
            print("Include posture task")
            self.posture_task = PostureTask(cost=self.config.posture_cost)
            self.tasks.append(self.posture_task)
        if self.config.damping_cost > 0:
            print("Include damping task")
            self.damping_task = DampingTask(cost=self.config.damping_cost)
            self.tasks.append(self.damping_task)

        # Set up the first target
        for task in self.tasks:
            if isinstance(task, FrameTask):
                task.set_target_from_configuration(self.configuration)
            if isinstance(task, PostureTask):
                task.set_target_from_configuration(self.configuration)

        return

    def _setup_barriers(self):
        self.barriers = []
        if self.config.self_collision_avoidance and len(self.robot.collision_model.collisionPairs) > 0:
            # TODO: Find a way to manage this
            print("Include self-collision avoidance barrier")
            self.barriers.append(
                SelfCollisionBarrier(
                    n_collision_pairs=len(self.robot.collision_model.collisionPairs),
                    gain=self.config.collision_gain,
                    safe_displacement_gain=self.config.safe_displacement_gain,
                    d_min=self.config.d_min,
                )
            )
        if self.config.box_barrier_gain > 0:
            print("Include Positions barriers")
            self.barriers.extend(
                [
                    PositionBarrier(
                        "L_ee",
                        p_min=self.left_eef_position_bounds.p_min,
                        p_max=self.left_eef_position_bounds.p_max,
                        gain=self.config.box_barrier_gain,
                        safe_displacement_gain=self.config.box_displacement_gain,
                    ),
                    PositionBarrier(
                        "R_ee",
                        p_min=self.right_eef_position_bounds.p_min,
                        p_max=self.right_eef_position_bounds.p_max,
                        gain=self.config.box_barrier_gain,
                        safe_displacement_gain=self.config.box_displacement_gain,
                    ),
                ]
            )

    def remove_position_barrier(self):
        self.barriers = [b for b in self.barriers if not isinstance(b, PositionBarrier)]

    def _setup_limits(self):
        self.limits = [self.configuration.model.configuration_limit, self.configuration.model.velocity_limit]
        if self.config.acceleration_limit is None:
            return
        self.limits.append(
            AccelerationLimit(
                self.robot.model,
                self.robot.model.velocityLimit * self.config.acceleration_limit,
            )
        )

    def _setup_visualizer(self, spawn_visualizer: bool):
        self.viz = start_viser_visualizer(self.robot, open=spawn_visualizer)
        self.viz.display(self.configuration.q)
        self._add_handle(self.left_task, scale=0.12)
        self._add_handle(self.right_task, scale=0.12)
        self._add_position_barrier_visual(self.viz.viewer, "L_ee", self.left_eef_position_bounds, color=(255, 120, 120))
        self._add_position_barrier_visual(
            self.viz.viewer, "R_ee", self.right_eef_position_bounds, color=(120, 160, 255)
        )

    def _add_ee_frame(self, frame_name: str, parent_joint_name: str, offset_x: float) -> None:
        if self.robot.model.existFrame(frame_name):
            return
        parent_joint_id = self.robot.model.getJointId(parent_joint_name)
        self.robot.model.addFrame(
            pin.Frame(
                frame_name,
                parent_joint_id,
                pin.SE3(np.eye(3), np.array([offset_x, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )

    def _add_handle(self, task: FrameTask, scale: float = 0.12):
        assert task.transform_target_to_world is not None, "Expected task.transform_target_to_world to be initialized"
        pose = task.transform_target_to_world.np
        assert self.viz is not None, "Visualizer must be initialized to add handle"
        handle = self.viz.viewer.scene.add_transform_controls(
            "/" + task.frame,
            position=pose[:3, 3],
            wxyz=Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True),
            scale=scale,
            line_width=3.0,
        )

        @handle.on_update
        def _on_update(evt: viser.TransformControlsEvent, task=task):
            target = task.transform_target_to_world
            target.translation = np.asarray(evt.target.position)
            target.rotation = Rotation.from_quat(evt.target.wxyz, scalar_first=True).as_matrix()

        return handle

    @staticmethod
    def _add_position_barrier_visual(
        viz: viser.ViserServer,
        frame_name: str,
        bounds: PositionBarrierBounds,
        color: tuple[int, int, int],
    ):
        scene = viz.scene
        center = 0.5 * (bounds.p_min + bounds.p_max)
        dimensions = bounds.p_max - bounds.p_min
        return scene.add_box(
            f"/barriers/{frame_name}",
            position=center,
            dimensions=dimensions,
            color=color,
            opacity=0.6,
            wireframe=True,
        )

    def get_targets_from_configuration(self) -> Targets:
        return Targets(
            left=self.configuration.get_transform_frame_to_world("L_ee"),
            right=self.configuration.get_transform_frame_to_world("R_ee"),
        )

    def get_targets(self) -> Targets:
        return Targets(
            left=self.left_task.transform_target_to_world,
            right=self.right_task.transform_target_to_world,
        )

    def set_targets(
        self,
        left: pin.SE3 | np.ndarray | None = None,
        right: pin.SE3 | np.ndarray | None = None,
    ) -> None:
        if left is not None:
            self.left_task.set_target(self._as_se3(left))
        if right is not None:
            self.right_task.set_target(self._as_se3(right))

    def step(self, dt: float = 0.01) -> np.ndarray:
        velocity = solve_ik(
            self.configuration,
            self.tasks,
            dt,
            solver=self.solver,
            safety_break=False,
            barriers=self.barriers,
            limits=self.limits,
        )
        q = pin.integrate(self.robot.model, self.configuration.q, velocity * dt)
        return q.copy()

    def solve(self, dt: float = 0.01, n_steps: int = 1) -> np.ndarray:
        q = self.configuration.q.copy()
        for _ in range(n_steps):
            q = self.step(dt=dt)
        return q

    def solve_dq(self, q: Optional[np.ndarray] = None, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Solve for dq given a configuration q, Returns the new configuration and the velocity."""
        if q is not None:
            self.configuration.update(q)
        velocity = solve_ik(
            self.configuration,
            self.tasks,
            dt,
            solver=self.solver,
            safety_break=False,
            barriers=self.barriers,
            limits=self.limits,
        )

        q = pin.integrate(self.robot.model, self.configuration.q, velocity * dt)
        return q, velocity

    def get_ee_poses(
        self,
        q: np.ndarray | None = None,
    ) -> Targets:
        """
        Return current L_ee and R_ee poses without mutating self.configuration.

        Args:
            q: Optional configuration to evaluate. If None, uses self.configuration.q.

        Returns:
            Targets with left/right EE poses in the world frame.
        """
        if q is None:
            q = self.configuration.q

        q = np.asarray(q).copy()
        data = self.robot.model.createData()

        pin.forwardKinematics(self.robot.model, data, q)
        pin.updateFramePlacements(self.robot.model, data)

        return Targets(
            left=data.oMf[self.left_ee_id].copy(),
            right=data.oMf[self.right_ee_id].copy(),
        )

    def compute_errors(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute the error and cost-weighted errors for each of the tasks and barrriers
        Returns:
            errors: dict mapping task/barrier name to its raw error
                (e.g. position error in meters)
            errors_norm: dict mapping task/barrier name to its cost-weighted error
                (e.g. position error times position cost)
        """
        errors = {}  # raw error
        errors_norm = {}  # error times the cost
        for task in self.tasks:
            if isinstance(task, FrameTask):
                error = task.compute_error(self.configuration)
                pos_error = np.linalg.norm(error[:3])
                ori_error = np.linalg.norm(error[3:])
                errors[f"{task.frame}_pos_error"] = pos_error
                errors[f"{task.frame}_ori_error"] = ori_error
                errors_norm[f"{task.frame}_pos_error"] = task.position_cost * pos_error
                errors_norm[f"{task.frame}_ori_error"] = task.orientation_cost * ori_error
            elif isinstance(task, PostureTask):
                error = task.compute_error(self.configuration)
                norm_error = np.linalg.norm(error)
                errors["posture_error"] = norm_error
                errors_norm["posture_error"] = task.cost * norm_error
            else:
                error = task.compute_error(self.configuration)
                norm_error = np.linalg.norm(error)
                errors[type(task).__name__ + "_error"] = norm_error
                errors_norm[type(task).__name__ + "_error"] = task.cost * norm_error

        for barrier in self.barriers:
            error = barrier.compute_barrier(self.configuration)
            norm_error = np.linalg.norm(error)
            errors[type(barrier).__name__ + "_error"] = norm_error
            errors_norm[type(barrier).__name__ + "_error"] = barrier.gain * norm_error

        return errors, errors_norm

    def get_closest_collision_pairs(
        self,
        q: np.ndarray | None = None,
        n: int = 20,
    ) -> list[tuple[float, str, str]]:
        """
        Return the n closest collision pairs and their distances.
        Useful for diagnosing self-collision barrier noise.

        Args:
            q: Configuration to evaluate. If None, uses self.configuration.q.
            n: Number of closest pairs to return.

        Returns:
            List of (distance, link_name_1, link_name_2) sorted by distance ascending.
        """
        if q is None:
            q = self.configuration.q

        q = np.asarray(q).copy()
        data = self.robot.model.createData()
        geom_data = pin.GeometryData(self.robot.collision_model)

        pin.forwardKinematics(self.robot.model, data, q)
        pin.updateGeometryPlacements(self.robot.model, data, self.robot.collision_model, geom_data)
        pin.computeDistances(self.robot.collision_model, geom_data)

        results = []
        for idx, pair in enumerate(self.robot.collision_model.collisionPairs):
            dist = geom_data.distanceResults[idx].min_distance
            name1 = self.robot.collision_model.geometryObjects[pair.first].name
            name2 = self.robot.collision_model.geometryObjects[pair.second].name
            results.append((dist, name1, name2))

        results.sort(key=lambda x: x[0])
        return results[:n]

    @staticmethod
    def _as_se3(T: pin.SE3 | np.ndarray) -> pin.SE3:
        """Utility function to convert a 4x4 numpy array to pin.SE3 if needed"""
        if isinstance(T, pin.SE3):
            return T
        T = np.asarray(T)
        if T.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform, got shape {T.shape}")
        return pin.SE3(T[:3, :3], T[:3, 3])

    @staticmethod
    def _default_position_bounds() -> PositionBarrierBounds:
        x_bounds = np.array([0.10, 0.40], dtype=float)
        z_bounds = np.array([-0.1, 0.3], dtype=float)
        y_bounds = np.array([-0.4, 0.4], dtype=float)
        return PositionBarrierBounds(
            p_min=np.array([x_bounds[0], y_bounds[0], z_bounds[0]], dtype=float),
            p_max=np.array([x_bounds[1], y_bounds[1], z_bounds[1]], dtype=float),
        )
