import numpy as np
import json
import os
from dataclasses import dataclass
import spatialmath as sm
from typing import Optional
import pinocchio as pin

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl import String_

from motion_tools.robot_gui import ReRunRobot

import rs_imle_policy.utils.transforms as transforms_utils
from rs_imle_policy.robots.base import BaseRobot
from rs_imle_policy.g1_arm_ik import G1ReducedPinkIK
from rs_imle_policy.unitree import G1_29_ArmController
from rs_imle_policy.inspire import Inspire_Controller_FTP
from rs_imle_policy.configs.g1_configs import G1IKConfigSim, G1IKConfigReal
import time


def publish_reset_category(category: int, publisher):  # Scene Reset signal for simulation
    msg = String_(data=str(category))
    publisher.Write(msg)


@dataclass
class RobotState:
    q: np.ndarray
    "Joint positions for the entire robot"
    q_arm: np.ndarray
    "Joint positions for the arms only"
    X_WL: sm.SE3
    "End-effector pose for the left arm in world frame"
    X_WR: sm.SE3
    "End-effector pose for the right arm in world frame"
    left_arm_state: np.ndarray
    "Concatenated position and orientation (6D) for the left arm end-effector [x,y,z,6d rotation]"
    right_arm_state: np.ndarray
    "Concatenated position and orientation (6D) for the right arm end-effector [x,y,z,6d rotation]"
    left_hand_state: np.ndarray
    "State of the left hand 6 dof"
    right_hand_state: np.ndarray
    "State of the right hand 6 dof"

    def build_low_level_state(self) -> np.ndarray:
        """Build a low-dimensional state representation for policy input."""
        return np.concatenate([self.left_arm_state, self.right_arm_state, self.left_hand_state, self.right_hand_state])


class G1RobotInterface(BaseRobot):
    def __init__(
        self,
        simulation: bool = True,
        visualizer: Optional[ReRunRobot] = None,
        ik_visualizer: Optional[ReRunRobot] = None,
    ):
        self.simulation = simulation
        dds_domain_id = 1 if simulation else 0
        if simulation:
            ChannelFactoryInitialize(id=dds_domain_id)  # dds domain id
        else:
            ChannelFactoryInitialize(
                id=dds_domain_id, networkInterface=os.environ["G1_NETWORK_INTERFACE"]
            )  # dds domain id
        print(dds_domain_id)
        ik_config = G1IKConfigSim() if simulation else G1IKConfigReal()
        self.controller = G1_29_ArmController(motion_mode=True, simulation_mode=simulation, sub_mode=False)
        self.hand_controller = Inspire_Controller_FTP()
        self.visualizer = visualizer
        self.ik_visualizer = ik_visualizer
        self.q0 = self.controller.get_current_dual_arm_q()
        if self.visualizer is not None:
            q0 = self.controller.get_current_motor_q()
            self.visualizer.log(q0)
        self.ik_solver = G1ReducedPinkIK(
            config=ik_config,
            visualize=False,
            spawn_visualizer=False,
            q0=self.q0.copy(),
        )
        breakpoint()
        pairs = self.ik_solver.get_closest_collision_pairs(n=20)
        for dist, a, b in pairs:
            print(f"{dist:.4f}m  {a}  <->  {b}")
        targets = self.ik_solver.get_targets_from_configuration()
        self.ik_solver.set_targets(targets.left, targets.right)
        self.init_arms()
        if simulation:
            self.reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            self.reset_pose_publisher.Init()

    def reset_sim(self):
        print("Resetting simulation to initial pose")
        if self.simulation:
            self.move_to_start()
            publish_reset_category(1, self.reset_pose_publisher)
        print("Real robot does not have reser sim method")

    def init_arms(self):
        self.ik_solver.remove_position_barrier()
        # Perform init sequence
        print("Initializing arms to starting configuration")
        with open("startup_g1.json", "r") as f:
            startup_config = json.load(f)
        waypoints = startup_config["waypoints"]

        l_poses = [pin.SE3(np.array(waypoint["left"]["matrix"]).reshape(4, 4)) for waypoint in waypoints]
        r_poses = [pin.SE3(np.array(waypoint["right"]["matrix"]).reshape(4, 4)) for waypoint in waypoints]

        # Check which waypoint is closest to the current configuration and start from there
        distances = []
        current_poses = self.ik_solver.get_ee_poses(self.controller.get_current_dual_arm_q())
        for l_waypoint, r_waypoint in zip(l_poses, r_poses):
            dist_l = current_poses.left.translation - l_waypoint.translation
            dist_r = current_poses.right.translation - r_waypoint.translation
            dist = np.linalg.norm(dist_l) + np.linalg.norm(dist_r)
            distances.append(dist)

        start_index = np.argmin(distances)

        l_poses.insert(start_index, current_poses.left)
        r_poses.insert(start_index, current_poses.right)

        for i in range(start_index, len(l_poses) - 1):
            l_start = l_poses[i]
            l_end = l_poses[i + 1]

            r_start = r_poses[i]
            r_end = r_poses[i + 1]

            for t in range(100):
                alpha = t / (100 - 1)

                l_interp = pin.SE3.Interpolate(l_start, l_end, alpha)
                r_interp = pin.SE3.Interpolate(r_start, r_end, alpha)

                self.ik_solver.set_targets(l_interp, r_interp)
                self.step_servo()
                time.sleep(0.01)

        self.ik_solver._setup_barriers()
        self.q0 = self.controller.get_current_dual_arm_q()

    def move_to_start(self, home_config=np.zeros(14)):
        q_sol = self.q0
        q_current = self.controller.get_current_dual_arm_q()

        while np.linalg.norm(q_sol - q_current) > 0.1:
            q_tauff = pin.rnea(
                self.ik_solver.robot.model,
                self.ik_solver.robot.data,
                q_sol,
                np.zeros(self.ik_solver.robot.model.nv),
                np.zeros(self.ik_solver.robot.model.nv),
            )
            self.controller.ctrl_dual_arm(q_sol, q_tauff)
            time.sleep(0.01)
            q_current = self.controller.get_current_dual_arm_q()
            print("Resetting arms, current error: ", np.linalg.norm(q_sol - q_current))

    def get_state(self) -> RobotState:
        """
        Get the current state of the robot, including joint positions and end-effector poses.
        Return:
            RobotState: A dataclass containing the robot's joint positions, end-effector poses, and low-level state representation.
        """
        q = self.controller.get_current_motor_q()
        q_arm = self.controller.get_current_dual_arm_q()

        poses = self.ik_solver.get_ee_poses(q_arm)
        X_WL = poses.left.homogeneous
        X_WR = poses.right.homogeneous

        pos_l, rot_l = transforms_utils.extract_robot_pos_orien(X_WL)
        pos_r, rot_r = transforms_utils.extract_robot_pos_orien(X_WR)

        left_hand_state, right_hand_state = self.hand_controller.get_state()

        return RobotState(
            q=q,
            q_arm=q_arm,
            X_WL=X_WL,
            X_WR=X_WR,
            left_arm_state=np.concatenate([pos_l, rot_l]),
            right_arm_state=np.concatenate([pos_r, rot_r]),
            left_hand_state=left_hand_state,
            right_hand_state=right_hand_state,
        )

    def set_ee_targets(self, left, right) -> None:
        self.ik_solver.set_targets(left, right)

    def step_servo(self, dt: float = 1 / 200) -> np.ndarray:
        pairs = self.ik_solver.get_closest_collision_pairs(n=1)
        for dist, a, b in pairs:
            print(f"{dist:.4f}m  {a}  <->  {b}")
        q_arm = self.controller.get_current_dual_arm_q()
        q = self.controller.get_current_motor_q()
        self.ik_solver.configuration.update(q_arm.copy())

        if self.visualizer is not None:
            self.visualizer.log(q)
            self.visualizer.log_pin_transform("left_ee", self.ik_solver.get_targets().left, parent_frame="world")
            self.visualizer.log_pin_transform("right_ee", self.ik_solver.get_targets().right, parent_frame="world")
            self.visualizer.log_pin_transform("pelvis", self.ik_solver.robot.data.oMf[1], parent_frame="world")

        q_sol = self.ik_solver.solve(dt=dt, n_steps=20)
        if self.ik_visualizer is not None:
            q_sol_full = q.copy()
            q_sol_full[15:29] = q_sol
            self.ik_visualizer.log(q_sol_full)

        q_tauff = pin.rnea(
            self.ik_solver.robot.model,
            self.ik_solver.robot.data,
            q_sol,
            np.zeros(self.ik_solver.robot.model.nv),
            np.zeros(self.ik_solver.robot.model.nv),
        )
        self.controller.ctrl_dual_arm(q_sol, q_tauff)
        return q_sol

    def get_gripper_state(self) -> float:
        print("Gripper state retrieval not implemented yet")
        return 0.0

    def close_gripper(self):
        print("Gripper control not implemented yet")

    def open_gripper(self):
        print("Gripper control not implemented yet")


if __name__ == "__main__":
    import rerun as rr

    rec = rr.RecordingStream("g1_robot_interface_test")
    rec.spawn()

    robot_visualizer = ReRunRobot.g1(rec, "g1_robot", "world")
    ik_visualizer = ReRunRobot.g1_debug(rec, "g1_ik_debug", "world")
    ik_visualizer.apply_color([1.0, 0.0, 0.0, 0.5])

    robot_interface = G1RobotInterface(simulation=False, visualizer=robot_visualizer, ik_visualizer=ik_visualizer)
