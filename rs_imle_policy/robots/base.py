from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from numpy.typing import NDArray


class BaseRobot(ABC):
    @abstractmethod
    def get_gripper_state(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def move_to_start(self, home_config: Optional[NDArray]) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_gripper(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def open_gripper(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> NDArray:
        raise NotImplementedError

    # @abstractmethod
    # def get_robot_state(self):
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def set_next_waypoints(self, translations, orientations, relative: bool = False) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def init_waypoint_motion(self) -> None:
    #     raise NotImplementedError
