import os
from frankx import Robot

with Robot(os.environ["PANDA_IP"]) as robot:
    robot.lock_brakes()
