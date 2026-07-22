"""Velocity-interface adapter for the Gear-Sonic ZMQ planner."""

from rs_imle_policy.configs.g1.locomotion import GearSonicConfig
from rs_imle_policy.controllers.g1.gear_sonic_interface import GearSonicInterface
from rs_imle_policy.controllers.g1.locomotion.commands import PlanarVelocity


class GearSonicVelocityInterface:
    def __init__(self, config: GearSonicConfig | None = None) -> None:
        self.config = config or GearSonicConfig()
        self._interface = GearSonicInterface(self.config)

    def start(self) -> None:
        self._interface.start()

    def send(self, command: PlanarVelocity) -> None:
        self._interface.set_velocity(command)

    def stop(self) -> None:
        if self._interface.is_running:
            self._interface.set_velocity(PlanarVelocity())

    def set_vr_poses(
        self,
        positions: tuple[float, ...],
        orientations: tuple[float, ...],
    ) -> None:
        self._interface.set_vr_poses(positions, orientations)

    def close(self) -> None:
        self._interface.close()

    def __enter__(self) -> "GearSonicVelocityInterface":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["GearSonicVelocityInterface"]
