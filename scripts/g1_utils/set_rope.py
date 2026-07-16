from typing import Literal

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


ELASTIC_BAND_TOPIC = "rt/elastic_band"


def main(
    command: Literal["enable", "disable"],
    domain_id: int = 1,
    network_interface: str = "lo",
) -> None:
    """Enable or disable the rope by publishing an elastic-band command."""
    ChannelFactoryInitialize(id=domain_id, networkInterface=network_interface)

    publisher = ChannelPublisher(ELASTIC_BAND_TOPIC, String_)
    publisher.Init()
    publisher.Write(String_(data=command))
    print(f"Published rope command {command!r} on {ELASTIC_BAND_TOPIC}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main, config=(tyro.conf.PositionalRequiredArgs,))
