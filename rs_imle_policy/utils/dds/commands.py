from typing import Literal

from rs_imle_policy.utils.dds.publisher import publish_string


RopeCommand = Literal["enable", "disable"]


def reset_sim(channel: str) -> None:
    """Reset the simulation through the configured DDS channel."""
    publish_string(channel, "-1")


def reset_dolly(channel: str="rt/reset_dolly") -> None:
    """Reset the dolly through the configured DDS channel."""
    publish_string(channel, "reset")


def set_rope(channel: str, command: RopeCommand) -> None:
    """Enable or disable the rope through the configured DDS channel."""
    publish_string(channel, command)
