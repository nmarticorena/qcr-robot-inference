"""Pose PID controller for G1 locomotion."""

from dataclasses import dataclass
import time

from loop_rate_limiters import RateLimiter
import numpy as np
import rerun as rr
import spatialmath as sm
import spatialmath.base as smb
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

from rs_imle_policy.configs.g1.locomotion import (
    BasePIDControllerConfig,
    PIDGainsConfig,
)
from rs_imle_policy.controllers.g1.locomotion.commands import (
    PlanarVelocity,
    create_velocity_interface,
)


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    @classmethod
    def from_message(cls, message: PoseStamped_) -> "Pose":
        pose = cls()
        pose.update_from_message(message)
        return pose

    def update_from_message(self, message: PoseStamped_) -> None:
        position = message.pose.position
        orientation = message.pose.orientation
        self.x, self.y, self.z = position.x, position.y, position.z
        self.w = orientation.w
        self.rx, self.ry, self.rz = orientation.x, orientation.y, orientation.z

    def to_se3(self) -> sm.SE3:
        transform = sm.SE3([self.x, self.y, self.z])
        transform.A[:3, :3] = smb.q2r([self.w, self.rx, self.ry, self.rz], order="sxyz")
        return transform

    def update_from_se3(self, transform: sm.SE3) -> None:
        quaternion = smb.r2q(transform.R, order="sxyz")
        self.x, self.y, self.z = (float(value) for value in transform.t)
        self.w, self.rx, self.ry, self.rz = (float(value) for value in quaternion)


class PIDAxis:
    def __init__(self, gains: PIDGainsConfig, name: str, log: bool = False) -> None:
        self.gains = gains
        self.name = name
        self.log = log
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        if self.gains.integral_limit > 0.0:
            self.integral = float(
                np.clip(
                    self.integral + error * dt,
                    -self.gains.integral_limit,
                    self.gains.integral_limit,
                )
            )
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error
        output = self.gains.kp * error + self.gains.ki * self.integral + self.gains.kd * derivative
        if self.log:
            rr.log(f"pid/{self.name}/error", rr.Scalars(error))
            rr.log(f"pid/{self.name}/output", rr.Scalars(output))
        return float(np.clip(output, -self.gains.output_limit, self.gains.output_limit))


class G1BaseController:
    """Follow a target pose with PID control through either walking backend."""

    def __init__(self, config: BasePIDControllerConfig | None = None) -> None:
        self.config = config or BasePIDControllerConfig()
        if self.config.hz <= 0.0:
            raise ValueError("controller frequency must be positive")
        self.command_interface = create_velocity_interface(self.config.locomotion)
        self.robot_pose = Pose()
        self.target_pose = Pose()
        self.have_robot_pose = False
        self.have_target_pose = False
        self.command_velocity = np.zeros(3, dtype=float)
        self.last_step_time = time.monotonic()
        self.started_at: float | None = None
        self.step_applied = False
        self.pid_x = PIDAxis(self.config.x_gains, "x", self.config.log)
        self.pid_y = PIDAxis(self.config.y_gains, "y", self.config.log)
        self.pid_yaw = PIDAxis(self.config.yaw_gains, "yaw", self.config.log)

        self.robot_pose_subscriber = ChannelSubscriber(self.config.dds.torso_pose_topic, PoseStamped_)
        self.robot_pose_subscriber.Init(self._on_robot_pose, 1)
        if self.config.target_step is None:
            self.target_pose_subscriber = ChannelSubscriber(self.config.dds.dolly_pose_topic, PoseStamped_)
            self.target_pose_subscriber.Init(self._on_target_pose, 1)

    def _on_robot_pose(self, message: PoseStamped_) -> None:
        self.robot_pose.update_from_message(message)
        self.have_robot_pose = True
        if self.config.target_step is not None and not self.have_target_pose:
            self.target_pose.update_from_message(message)
            self.have_target_pose = True
            self.started_at = time.monotonic()

    def _on_target_pose(self, message: PoseStamped_) -> None:
        self.target_pose.update_from_message(message)
        self.have_target_pose = True

    def _apply_target_step(self, now: float) -> None:
        step = self.config.target_step
        if step is None or self.step_applied or self.started_at is None:
            return
        if now - self.started_at < step.delay:
            return
        target = self.robot_pose.to_se3() * sm.SE3.Trans(step.x, step.y, 0.0) * sm.SE3.Rz(step.yaw)
        self.target_pose.update_from_se3(target)
        self.step_applied = True

    def _limit_acceleration(self, target: PlanarVelocity, dt: float) -> PlanarVelocity:
        requested = np.asarray(target.as_tuple())
        acceleration = np.asarray(
            [
                self.config.limits.max_ax,
                self.config.limits.max_ay,
                self.config.limits.max_alpha,
            ]
        )
        delta = np.clip(
            requested - self.command_velocity,
            -acceleration * dt,
            acceleration * dt,
        )
        self.command_velocity += delta
        return PlanarVelocity(*self.command_velocity)

    def step(self) -> None:
        now = time.monotonic()
        dt = now - self.last_step_time
        self.last_step_time = now
        if dt <= 0.0 or dt > 1.0:
            for pid in (self.pid_x, self.pid_y, self.pid_yaw):
                pid.reset()
            dt = 1.0 / self.config.hz
        self._apply_target_step(now)
        relative = self.robot_pose.to_se3().inv() * self.target_pose.to_se3()
        yaw_error = relative.rpy(order="zyx")[2]
        requested = PlanarVelocity(
            self.pid_x.update(relative.t[0], dt) if self.config.enable_x else 0.0,
            self.pid_y.update(relative.t[1], dt) if self.config.enable_y else 0.0,
            self.pid_yaw.update(yaw_error, dt) if self.config.enable_yaw else 0.0,
        )
        self.command_interface.send(self._limit_acceleration(requested, dt))

    def run(self) -> None:
        self.command_interface.start()
        rate = RateLimiter(self.config.hz)
        try:
            while not (self.have_robot_pose and self.have_target_pose):
                time.sleep(0.1)
            while True:
                self.step()
                rate.sleep()
        finally:
            self.command_interface.close()


def main(config: BasePIDControllerConfig) -> None:
    ChannelFactoryInitialize(
        config.dds.domain_id,
        config.dds.network_interface,
    )
    if config.log:
        rr.init("g1_base_controller", spawn=True)
    G1BaseController(config).run()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(BasePIDControllerConfig))


__all__ = [
    "G1BaseController",
    "PIDAxis",
    "Pose",
]
