import numpy as np
from rs_imle_policy.robots.g1 import G1RobotInterface

robot_interface = G1RobotInterface(simulation=False, init_arms = False)
home = np.zeros(14)
home[3] = np.pi/2
home[3+7] = np.pi/2

robot_interface.move_to_start(home)
