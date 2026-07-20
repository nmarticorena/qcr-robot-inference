import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from rs_imle_policy.utils.dds import reset_dolly



def main(
    domain_id: int = 1,
    network_interface: str = "lo",
    repeat: int = 1,
) -> None:
    ChannelFactoryInitialize(id=domain_id, networkInterface=network_interface)

    idx = 0
    while idx < repeat:
        reset_dolly()
        time.sleep(0.001)
        idx += 1


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
