"""DDS SportMode velocity transport."""

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from rs_imle_policy.configs.g1.locomotion import SportModeConfig
from rs_imle_policy.controllers.g1.locomotion.commands import PlanarVelocity


class SportModeVelocityInterface:
    """Publish planar commands in the simulator/Unitree SportMode format."""

    def __init__(self, config: SportModeConfig | None = None) -> None:
        self.config = config or SportModeConfig()
        self._publisher = ChannelPublisher(self.config.command_topic, String_)
        self._publisher.Init()

    def start(self) -> None:
        """SportMode DDS needs no explicit start pulse."""

    def send(self, command: PlanarVelocity) -> None:
        vx, vy, omega = command.as_tuple()
        self._publisher.Write(String_(f"{vx} {vy} {omega} {self.config.command_height}"))

    def stop(self) -> None:
        self.send(PlanarVelocity())

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "SportModeVelocityInterface":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["SportModeVelocityInterface"]
