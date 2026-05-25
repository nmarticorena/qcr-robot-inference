from franky import Robot, CartesianMotion, Affine, ReferenceType, RealtimeConfig
import os

robot = Robot(os.environ["PANDA_IP"], realtime_config=RealtimeConfig.Ignore)  # Replace this with your robot's IP
robot.recover_from_errors()

# Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
robot.relative_dynamics_factor = 0.05

# Move the robot 20cm along the relative X-axis of its end-effector
motion = CartesianMotion(Affine([-0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion)
