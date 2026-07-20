from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from rs_imle_policy.utils.dds import RopeCommand, set_rope




def main(
    command: RopeCommand,
    domain_id: int = 1,
    network_interface: str = "lo",
) -> None:
    """Enable or disable the rope by publishing an elastic-band command."""
    ChannelFactoryInitialize(id=domain_id, networkInterface=network_interface)
    set_rope(command)
    print(f"Published rope command {command!r}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main, config=(tyro.conf.PositionalRequiredArgs,))
