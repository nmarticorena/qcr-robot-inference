from enum import Enum
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from frankx import Gripper, Robot, JointMotion, Waypoint, WaypointMotion, Affine
from panda_py import Panda, libfranka

from rs_imle_policy.utils import transforms
from rs_imle_policy.robots.base import BaseRobot


# Constants TODO: Decide where is better to have these
DEFAULT_ROBOT_IP = "172.16.0.2"
DEFAULT_GRIPPER_SPEED = 0.1
DEFAULT_GRIPPER_FORCE = 40
GRIPPER_OPEN_WIDTH = 0.08
DEFAULT_DYNAMIC_REL = 0.4
CARTESIAN_IMPEDANCE = [400.0, 400.0, 400.0, 40.0, 40.0, 40.0]
ACCEL_REL = 0.1
JERK_REL = 0.1


class GripperState(Enum):
    """Enumeration of gripper states."""

    OPEN = 0
    CLOSED = 1


class FrankxRobot(BaseRobot):
    """High-level interface for Franka Panda robot control.

    This class provides methods for robot motion control, gripper operations,
    and state monitoring using the frankx library.

    Attributes:
        gripper: Gripper controller instance
        robot: Robot controller instance
        gripper_state: Current state of the gripper
        move_async: Async motion handle
        motion: Current motion instance
        X_BE: End-effector pose transformation matrix
        pos: Current end-effector position
        rot: Current end-effector rotation matrix
    """

    def __init__(
        self,
        ip: str = DEFAULT_ROBOT_IP,
        dynamic_rel: float = DEFAULT_DYNAMIC_REL,
        dry_run: bool = False,
    ):
        """Initialize the robot controller.

        Args:
            ip: Robot IP address
            dynamic_rel: Dynamic scaling factor for robot motion
        """

        self.robot = Robot(ip, dynamic_rel=dynamic_rel, repeat_on_error=True)

        self.robot.recover_from_errors()
        self.gripper = Gripper(fci_ip=ip, speed=DEFAULT_GRIPPER_SPEED)
        self.gripper.open(True)
        self.gripper_state = GripperState.OPEN

        self.dry_run = dry_run
        self.move_async = None
        self.motion = None

        state = self.robot.read_once()
        self.X_BE = np.array(state.O_T_EE).reshape(4, 4, order="F")

        self.pos = self.X_BE[:3, 3]
        self.rot = self.X_BE[:3, :3]

    def get_gripper_width(self) -> float:
        """Get current gripper width.

        Returns:
            Current gripper width in meters
        """
        return self.gripper.width()

    def initialize_cartesian_impedance(self):
        """Initialize Cartesian impedance control parameters."""
        self.robot.set_cartesian_impedance(CARTESIAN_IMPEDANCE)
        self.robot.accel_rel = ACCEL_REL
        self.robot.jerk_rel = JERK_REL

    def move_to_start(self, home_config: Optional[NDArray]):
        """Move robot to starting configuration.

        Args:
            home_config: Joint configuration for home position, or None to skip
        """
        if home_config is not None:
            self.robot.move(JointMotion(target=home_config))
        self.open_gripper()

    def close_gripper(self):
        """Close the gripper if not already closed."""
        print("Closing gripper")
        if self.gripper_state != GripperState.CLOSED:
            self.gripper.close(blocking=False)
            self.gripper_state = GripperState.CLOSED

    def open_gripper(self):
        """Open the gripper if not already open."""
        if self.gripper_state != GripperState.OPEN:
            self.gripper.open(blocking=False)
            self.gripper_state = GripperState.OPEN

    def stop_motion(self, release: bool = True):
        """Stop current robot motion.

        Args:
            release: If True, open the gripper after stopping
        """
        if self.motion is not None:
            self.motion.finish()
        if release:
            self.open_gripper()

    def get_state(self) -> NDArray:
        """Get current robot state.

        Returns state vector containing end-effector position, rotation (6D),
        and gripper width.

        Returns:
            NDArray: shape (10,), [x, y, z, r1, r2, r3, r4, r5, r6, gripper_width]

        Raises:
            RuntimeError: If motion has not been initialized
        """
        if self.motion is None:
            raise RuntimeError("Motion not initialized")
        state = self.motion.get_robot_state()

        X_BE = np.array(state.O_T_EE).reshape(4, 4, order="F")

        rot = transforms.matrix_to_rotation_6d(X_BE[:3, :3])
        pos = X_BE[:3, 3]
        width = self.get_gripper_width()

        # Store values for further use
        self.X_BE = X_BE
        self.pos = pos
        self.rot = X_BE[:3, :3]

        return np.concatenate([pos, rot, [width]])

    def get_robot_state(self):
        """Get the full robot state from the motion controller.

        Returns:
            Robot state object from frankx
        """
        if self.motion is None:
            raise RuntimeError("Motion not initialized")
        return self.motion.get_robot_state()

    def set_next_waypoints(self, translations, orientations, relative: bool = False):
        """Set the next waypoints for the robot to follow.

        Args:
            translations: Array of shape (N, 3) containing target positions
            orientations: Array of shape (N, 4) in quaternion format [w, x, y, z]
            relative: If True, waypoints are relative to current pose (not yet implemented)
        """
        assert self.motion is not None, "Motion not initialized"
        ref = Waypoint.Relative if relative else Waypoint.Absolute

        waypoints = [
            Waypoint(Affine(trans[0], trans[1], trans[2], q[0], q[1], q[2], q[3]), ref)
            for trans, q in zip(translations, orientations)
        ]

        if self.dry_run:
            return
        self.motion.set_next_waypoints(waypoints)

    def init_waypoint_motion(self):
        """Initialize waypoint motion controller.

        Creates a waypoint motion starting from the current robot pose and
        starts asynchronous motion execution.
        """
        current_pose = self.robot.current_pose()
        self.motion = WaypointMotion(
            [Waypoint(affine=current_pose)],
            return_when_finished=False,
        )
        self.move_async = self.robot.move_async(self.motion)


class PandaPyRobot:
    """High-level interface for Franka Panda robot control using pandapy.

    This class use the extra controllers available in panda_py such as the
    teaching_mode controller.

    Attributes:
        gripper: Gripper controller instance
        robot: PandaPy Robot controller instance
        gripper_state: Current state of the gripper
        X_BE: End-effector pose transformation matrix
        pos: Current end-effector position
        rot: Current end-effector rotation matrix
    """

    def __init__(
        self,
        ip: str = DEFAULT_ROBOT_IP,
        dynamic_rel: float = DEFAULT_DYNAMIC_REL,
        dry_run: bool = False,
    ):
        """Initialize the PandaPy robot controller.

        Args:
            ip: Robot IP address
            dynamic_rel: Dynamic scaling factor for robot motion
        """

        self.robot = Panda(hostname=ip)

        self.robot.get_robot().automatic_error_recovery()
        self.gripper = libfranka.Gripper(ip)
        self.gripper.move(GRIPPER_OPEN_WIDTH, DEFAULT_GRIPPER_SPEED)
        self.gripper_state = GripperState.OPEN

        self.dry_run = dry_run

        state = self.robot.get_robot().read_once()
        self.X_BE = np.array(state.O_T_EE).reshape(4, 4, order="F")

        self.pos = self.X_BE[:3, 3]
        self.rot = self.X_BE[:3, :3]

    def get_gripper_width(self) -> float:
        """Get current gripper width.

        Returns:
            Current gripper width in meters
        """
        return self.gripper.read_once().width

    def move_to_start(self, home_config: Optional[NDArray]):
        """Move robot to starting configuration.

        Args:
            home_config: Joint configuration for home position, or None to skip
        """
        self.robot.teaching_mode(False)
        if home_config is not None:
            self.robot.move_to_joint_position(home_config)
        self.open_gripper()

    def close_gripper(self):
        """Close the gripper if not already closed."""
        if self.gripper_state != GripperState.CLOSED:
            self.gripper.grasp(0.0, DEFAULT_GRIPPER_SPEED, DEFAULT_GRIPPER_FORCE)
            self.gripper_state = GripperState.CLOSED

    def open_gripper(self):
        """Open the gripper if not already open."""
        if self.gripper_state != GripperState.OPEN:
            self.gripper.move(GRIPPER_OPEN_WIDTH, DEFAULT_GRIPPER_SPEED)
            self.gripper_state = GripperState.OPEN

    def get_state(self):
        """Get Current robot state.

        Returns state vector containing end-effector position, rotation (6D),
        and gripper width.

        Returns:
            NDArray: shape (10,), [x, y, z, r1, r2, r3, r4, r5, r6, gripper_width]
        """
        state = self.robot.get_state()

        X_BE = np.array(state.O_T_EE).reshape(4, 4, order="F")

        rot = transforms.matrix_to_rotation_6d(X_BE[:3, :3])
        pos = X_BE[:3, 3]
        width = self.get_gripper_width()

        # Store values for further use
        self.X_BE = X_BE
        self.pos = pos
        self.rot = X_BE[:3, :3]

        return np.concatenate([pos, rot, [width]])

    def get_robot_state(self):
        """Get the full robot state

        Returns:
            Robot state object from panda_py
        """
        return self.robot.get_state()

    def set_next_waypoints(self, translations, orientations, relative: bool = False):
        """Set the next waypoints for the robot to follow.

        Args:
            translations: Array of shape (N, 3) containing target positions
            orientations: Array of shape (N, 4) in quaternion format [w, x, y, z]
            relative: If True, waypoints are relative to current pose (not yet implemented)
        """
        print(
            "Waypoints not implemented for pandapy controller yet, this interface is mainly for",
            "using the teaching mode controller (gravity compensation) for inference using ",
            "the dry run option. If you want to use the waypoint interface, please use the ",
            "FrankxRobot controller instead.",
        )
        return

    def init_waypoint_motion(self):
        """Initialize waypoint motion controller.

        Creates a waypoint motion starting from the current robot pose and
        starts asynchronous motion execution.
        """
        self.robot.teaching_mode(True)
