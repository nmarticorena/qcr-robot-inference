"""
IK solver for the G1 Arms, it relies on Pink, and pinnochio
the collision geometry where approximated by convex hulls


"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pinocchio as pin
import qpsolvers
import viser
from scipy.spatial.transform import Rotation

import pink
from pink import solve_ik
from pink.barriers import SelfCollisionBarrier
from pink.tasks import FrameTask, PostureTask, DampingTask
from pink.utils import process_collision_pairs
from pink.visualization import start_viser_visualizer

from rs_imle_policy.utils.collision_utils import force_convex_collision_geometry
from rs_imle_policy.utils.urdf_utils import (
    BODY_JOINTS,
    INSPIRE_FTP_HAND_JOINTS,
    DEX3_HAND_JOINTS,
    INSPIRE_DFQ_HAND_JOINTS,
)


DEFAULT_LOCKED_JOINTS = (
    BODY_JOINTS + INSPIRE_FTP_HAND_JOINTS + DEX3_HAND_JOINTS + INSPIRE_DFQ_HAND_JOINTS
)


@dataclass
class Targets:
    left: pin.SE3
    right: pin.SE3


class G1ReducedPinkIK:
    def __init__(
        self,
        urdf_path: str,
        mesh_dirs: Iterable[str],
        *,
        srdf_path: str | None = None,
        locked_joints: Iterable[str] = DEFAULT_LOCKED_JOINTS,
        ee_offset: float = 0.05,
        use_free_flyer: bool = False,
        visualize: bool = False,
        enable_self_collision: bool = True,
        collision_top_k: int = 10,
        spawn_visualizer: bool = True,
        q0: np.ndarray | None = None,
        pos_cost: float | Sequence[float] = 20.0,
    ):
        self.urdf_path = urdf_path
        self.mesh_dirs = list(mesh_dirs)
        self.srdf_path = srdf_path
        self.visualize = visualize
        self.enable_self_collision = enable_self_collision
        self.collision_top_k = collision_top_k

        root_joint = pin.JointModelFreeFlyer() if use_free_flyer else None
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path,
            self.mesh_dirs,
            root_joint,
        )

        self.robot.collision_model.addAllCollisionPairs()
        converted = force_convex_collision_geometry(self.robot)
        print(f"Converted {converted} collision geometries to convex hulls")

        if srdf_path and os.path.exists(srdf_path):
            pin.removeCollisionPairs(
                self.robot.model, self.robot.collision_model, srdf_path
            )
            self.robot.collision_data = process_collision_pairs(
                self.robot.model,
                self.robot.collision_model,
                srdf_path,
            )
        else:
            self.robot.collision_data = pin.GeometryData(self.robot.collision_model)

        joint_ids_to_lock = set(
            [self.robot.model.getJointId(name) for name in locked_joints]
        )

        reference_configuration = pin.neutral(self.robot.model)
        self.robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joint_ids_to_lock,
            reference_configuration=reference_configuration,
        )
        if (
            not hasattr(self.robot, "collision_data")
            or self.robot.collision_data is None
        ):
            self.robot.collision_data = pin.GeometryData(self.robot.collision_model)

        self._add_ee_frame("L_ee", "left_wrist_yaw_joint", ee_offset)
        self._add_ee_frame("R_ee", "right_wrist_yaw_joint", ee_offset)

        # Recreate data after adding frames, otherwise model/data go out of sync
        self.robot.data = self.robot.model.createData()

        self.left_ee_id = self.robot.model.getFrameId("L_ee")
        self.right_ee_id = self.robot.model.getFrameId("R_ee")

        # if self.robot.model.nq >= 7:
        #     q0[2] = 0.72
        #     q0[6] = 1.0
        #
        if q0 is None:
            q0 = pin.neutral(self.robot.model)
        self.configuration = pink.Configuration(
            self.robot.model,
            self.robot.data,
            q0,
            collision_model=self.robot.collision_model,
            collision_data=self.robot.collision_data,
        )

        self.left_task = FrameTask(
            "L_ee",
            position_cost=pos_cost,
            orientation_cost=5.0,
        )
        self.right_task = FrameTask(
            "R_ee",
            position_cost=pos_cost,
            orientation_cost=5.0,
        )
        self.posture_task = PostureTask(cost=1e-2)
        self.damping_task = DampingTask(
            cost=1e-1,  # [cost] * [s] / [rad]
        )

        # self.tasks = [self.left_task, self.right_task, self.posture_task, self.damping_task]
        self.tasks = [
            self.left_task,
            self.right_task,
            self.damping_task,
            self.posture_task,
        ]
        for task in self.tasks:
            if isinstance(task, FrameTask):
                task.set_target_from_configuration(self.configuration)
            if isinstance(task, PostureTask):
                task.set_target_from_configuration(self.configuration)

        self.barriers = []
        if enable_self_collision and len(self.robot.collision_model.collisionPairs) > 0:
            self.barriers = [
                SelfCollisionBarrier(
                    n_collision_pairs=10,
                    gain=5.0,
                    safe_displacement_gain=0.1,
                    d_min=0.005,
                )
            ]

        self.solver = (
            "daqp"
            if "daqp" in qpsolvers.available_solvers
            else qpsolvers.available_solvers[0]
        )
        print(self.solver)

        self.viz = None
        self.debug_viewer = None
        if visualize:
            self.viz = start_viser_visualizer(self.robot, open=spawn_visualizer)
            self.viz.display(self.configuration.q)
            self._add_handle(self.left_task, scale=0.12)
            self._add_handle(self.right_task, scale=0.12)

    def _add_ee_frame(
        self, frame_name: str, parent_joint_name: str, offset_x: float
    ) -> None:
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
        pose = task.transform_target_to_world.np
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
            target.rotation = Rotation.from_quat(
                evt.target.wxyz, scalar_first=True
            ).as_matrix()

        return handle

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
            damping=1e-2,
            safety_break=False,
            barriers=self.barriers,
        )
        q = pin.integrate(self.robot.model, self.configuration.q, velocity * dt)
        return q.copy()

    def solve(self, dt: float = 0.01, n_steps: int = 1) -> np.ndarray:
        q = self.configuration.q.copy()
        for _ in range(n_steps):
            q = self.step(dt=dt)
        return q

    def solve_dq(
        self, left: pin.SE3 | np.ndarray, right: pin.SE3 | np.ndarray
    ) -> np.ndarray:
        self.set_targets(left=left, right=right)
        velocity = solve_ik(
            self.configuration,
            self.tasks,
            0.01,
            solver=self.solver,
            damping=1e-2,
            safety_break=False,
            barriers=self.barriers,
        )

        return velocity

    @staticmethod
    def _as_se3(T: pin.SE3 | np.ndarray) -> pin.SE3:
        if isinstance(T, pin.SE3):
            return T
        T = np.asarray(T)
        if T.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform, got shape {T.shape}")
        return pin.SE3(T[:3, :3], T[:3, 3])
