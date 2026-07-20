from typing import Literal

from rs_imle_policy.utils.dds.publisher import publish_string


RopeCommand = Literal["enable", "disable"]


def reset_sim(channel: str = "rt/reset_category") -> None:
    """Reset the simulation through the configured DDS channel."""
    publish_string(channel, "-1")


def reset_dolly(channel: str="rt/reset_dolly") -> None:
    """Reset the dolly through the configured DDS channel."""
    publish_string(channel, "reset")


def set_rope(command: RopeCommand, channel: str = "rt/elastic_band") -> None:
    """Enable or disable the rope through the configured DDS channel."""
    publish_string(channel, command)
