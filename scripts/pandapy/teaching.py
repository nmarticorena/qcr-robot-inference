import panda_py
import os

robot = panda_py.Panda(os.environ["PANDA_IP"])

robot.teaching_mode(True)
while True:
    pass
