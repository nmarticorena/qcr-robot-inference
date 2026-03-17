from typing import Optional
from dataclasses import dataclass
import numpy as np

"""
Configurations for the G1 Pink ik solver
if any value is 0.0 the task will not be added
"""


@dataclass(frozen=True)
class PositionBarrierBounds:
    p_min: np.ndarray
    p_max: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_min", np.asarray(self.p_min, dtype=float).reshape(3))
        object.__setattr__(self, "p_max", np.asarray(self.p_max, dtype=float).reshape(3))
        if np.any(self.p_min >= self.p_max):
            raise ValueError(f"Expected p_min < p_max, got p_min={self.p_min}, p_max={self.p_max}")


@dataclass
class G1IKConfig:
    urdf_path: str = "assets/g1.urdf"
    # Path for urdf
    srdf_path: str = "assets/g1_arms.srdf"
    # Path for srdf

    ee_offset: float = 0.05
    # Displacement between the wrist and the centre of the hands

    # Tasks costs
    pos_cost: float = 20.0
    # Cost of position error
    ori_cost: float = 5.0
    # Cost of rotation error
    posture_cost: float = 1e-5
    # Cost of deviating from current posture
    damping_cost: float = 1e-3
    # Cost for daming term

    # Barriers
    self_collision_avoidance: bool = True
    # Whether to add self-collision avoidance task
    box_barrier_gain: float = 10.0
    # Gain for box barrier task
    box_displacement_gain: float = 1.0
    # Gain for box displacement task

    # Limits
    acceleration_limit: Optional[float] = 10.0
    # Maximum acceleration for the joints as a multipler of the max velocity

    # QP configs
    solver: str = "daqp"
