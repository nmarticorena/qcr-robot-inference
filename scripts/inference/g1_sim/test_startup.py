import json
import numpy as np
from motion_tools.robot_gui import ReRunRobot
import rerun as rr
from pinocchio import SE3
import time

rec = rr.RecordingStream("rerun_g1_startup")
rec.spawn()
g1_robot = ReRunRobot.g1(rec, "g1_robot", "pelvis")


with open("startup_g1.json", "r") as f:
    startup_config = json.load(f)

waypoints = startup_config["waypoints"]

for i in range(len(waypoints) - 1):
    start = waypoints[i]
    end = waypoints[i + 1]

    l_start = SE3(np.array(start["left"]["matrix"]).reshape(4, 4))
    l_end = SE3(np.array(end["left"]["matrix"]).reshape(4, 4))

    r_start = SE3(np.array(start["right"]["matrix"]).reshape(4, 4))
    r_end = SE3(np.array(end["right"]["matrix"]).reshape(4, 4))

    for t in range(1000):
        alpha = t / 1000
        l_interp = SE3.Interpolate(l_start, l_end, alpha)
        r_interp = SE3.Interpolate(r_start, r_end, alpha)
        g1_robot.log_pin_transform("left_arm", l_interp)
        g1_robot.log_pin_transform("right_arm", r_interp)
        time.sleep(0.001)
