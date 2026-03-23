from rs_imle_policy.unitree import G1_29_ArmController
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from motion_tools.robot_gui import ReRunRobot
import rerun as rr

rec = rr.RecordingStream("g1_arm_controller_test")
rec.spawn()


robot_gui = ReRunRobot.g1(rec)

ChannelFactoryInitialize(id=1)  # dds domain id
controller = G1_29_ArmController(motion_mode=False, simulation_mode=True)

while True:
    q = controller.get_current_motor_q()
    robot_gui.log(q)
    controller.ctrl_dual_arm(np.random.rand(14), np.zeros(14))  # Todo the tauff is the feedforward torque
