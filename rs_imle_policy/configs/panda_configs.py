from typing import List
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SinglePandaConfig:
    """
    Config for frankx/pandanpy controller to keep the same args over all experiments
    """
    robot_ip:str = "172.16.0.2"

    # Controller
    cartesian_impedance:List[float] = field(
        default_factory=lambda: [400.0, 400.0, 400.0, 40.0, 40.0, 40.0]
    )
    dynamic_rel:float = 0.2
    accel_rel:float = 0.1
    jerk_rel:float = 0.01
    repeat_on_error: bool = True

    # Stop behaviour
    stop_duration_s: float = 1.
    stop_rate_hz: float = 10
    
    # Gripper
    gripper_speed:float = 0.8 # [m/s], Actually it cannot go that fast but just to have it as fast as possible
    gripper_open_width:float = 0.08 #[m]
    gripper_force: float = 40. # [Nm]

    # Task specific
    default_q: List[float] = field(
        default_factory=lambda: [
            0.0,
            -np.pi / 4,
            0.0,
            -3 * np.pi / 4,
            0.0,
            np.pi / 2,
            np.pi / 4,
        ]
    )

    # Policy related
    action_hz:float = 10. # Desire frecuency between waypoints
    gripper_close_th:float = 0.5 # th for consider the binnary action of closing is executed 
    progress_complete_th:float = 0.9 # th for consider the progress action is consider as successful


