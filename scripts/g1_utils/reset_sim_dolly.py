from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


RESET_TOPIC = "rt/reset_dolly"


def main(
    domain_id: int = 1,
    network_interface: str = "lo",
) -> None:
    ChannelFactoryInitialize(id=domain_id, networkInterface=network_interface)

    publisher = ChannelPublisher(RESET_TOPIC, String_)
    publisher.Init()
    publisher.Write(String_(data="reset"))
    print(f"Published dolly reset on {RESET_TOPIC}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
