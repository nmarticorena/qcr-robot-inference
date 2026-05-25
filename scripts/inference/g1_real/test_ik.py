import os
import time

import numpy as np
import pinocchio as pin
import rerun as rr
from loop_rate_limiters import RateLimiter
from motion_tools.robot_gui import ReRunRobot
from teleimager.image_client import ImageClient

# for simulation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from rs_imle_policy.g1_arm_ik import G1ReducedPinkIK
from rs_imle_policy.unitree import G1_29_ArmController

img_client = ImageClient(os.environ["G1_IP"], request_bgr=True)


ChannelFactoryInitialize(networkInterface=os.environ["G1_NETWORK_INTERFACE"], id=0)  # dds domain id


def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)


reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
reset_pose_publisher.Init()
publish_reset_category(1, reset_pose_publisher)


rec = rr.RecordingStream("g1_arm_controller_test")
controller = G1_29_ArmController(motion_mode=False, simulation_mode=False)
frecuecy = 200
rec.spawn()

ik = G1ReducedPinkIK(
    visualize=True,
    spawn_visualizer=True,
)


targets = ik.get_targets_from_configuration()
ik.set_targets(targets.left, targets.right)

robot_gui = ReRunRobot.g1(rec, target_frame="pelvis")
robot_sol = ReRunRobot.g1_debug(rec, target_frame="pelvis")
robot_sol.apply_color([0, 1, 0, 0.5])


ti = time.time()
while time.time() - ti < 0.05:
    pass

rate = RateLimiter(frecuecy, warn=False)
while True:
    target = ik.get_targets()
    robot_sol.log_pin_transform("left_ee", target.left)
    robot_sol.log_pin_transform("right_ee", target.right)
    ti = time.time()
    q = controller.get_current_motor_q()
    q_arm = controller.get_current_dual_arm_q()

    robot_gui.rec.log("state/arm_q", rr.Scalars(q_arm.copy()))
    q_left_arm = q_arm[:7]
    q_right_arm = q_arm[7:14]

    dq_arm = controller.get_current_dual_arm_dq()
    ik.configuration.update(q_arm.copy())
    ik.viz.display(q_arm.copy())
    q_sol = ik.solve(dt=1 / frecuecy, n_steps=1)

    robot_gui.rec.log("state/ik_q_sol", rr.Scalars(q_sol.copy()))
    full_q_sol = np.zeros(29)
    full_q_sol[15:29] = q_sol
    robot_sol.log(full_q_sol)
    robot_gui.log(q.copy())

    q_tauff = pin.rnea(
        ik.robot.model,
        ik.robot.data,
        q_sol,
        np.zeros(ik.robot.model.nv),
        np.zeros(ik.robot.model.nv),
    )
    controller.ctrl_dual_arm(q_sol, q_tauff)
    robot_gui.rec.log(
        "cameras/head_frame",
        rr.EncodedImage(contents=img_client.get_head_frame().jpg, media_type="image/jpeg"),
    )

    rate.sleep()
