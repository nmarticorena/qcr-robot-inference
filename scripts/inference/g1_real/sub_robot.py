import rerun as rr
import os
from motion_tools.robot_gui import ReRunRobot
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from rs_imle_policy.unitree import G1_29_ArmController

rec = rr.RecordingStream("g1_arm_controller_test")
rec.spawn()


robot_gui = ReRunRobot.g1(rec)

# TODO: Add network interface as
ChannelFactoryInitialize(networkInterface=os.environ["G1_NETWORK_INTERFACE"], id=0)  # dds domain id
controller = G1_29_ArmController(motion_mode=False, simulation_mode=False, sub_mode=True)
# breakpoint()

while True:
    q = controller.get_current_motor_q()

    robot_gui.log(q)
    q_arm = controller.get_current_dual_arm_q()
    print(q_arm)
    # controller.ctrl_dual_arm(
    #     np.random.rand(14), np.zeros(14)
    # )  # Todo the tauff is the feedforward torque
