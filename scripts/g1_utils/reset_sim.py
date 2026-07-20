from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from rs_imle_policy.utils.dds import reset_sim




def main(
    domain_id: int = 1,
    network_interface: str = "lo",
) -> None:
    ChannelFactoryInitialize(id=domain_id, networkInterface=network_interface)
    reset_sim()
    print("Done")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
