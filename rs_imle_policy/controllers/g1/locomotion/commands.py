"""Common velocity command types and walking-backend factory."""

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from rs_imle_policy.configs.g1.locomotion import LocomotionConfig, WalkingBackend


@dataclass(frozen=True)
class PlanarVelocity:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.vx, self.vy, self.omega)):
            raise ValueError("velocity values must be finite")

    def as_tuple(self) -> tuple[float, float, float]:
        return self.vx, self.vy, self.omega


@runtime_checkable
class VelocityInterface(Protocol):
    def start(self) -> None: ...

    def send(self, command: PlanarVelocity) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def create_velocity_interface(config: LocomotionConfig) -> VelocityInterface:
    """Create the selected walking backend without leaking backend arguments."""

    if config.backend is WalkingBackend.SPORT_MODE:
        from rs_imle_policy.controllers.g1.locomotion.sport_mode import (
            SportModeVelocityInterface,
        )

        return SportModeVelocityInterface(config.sport_mode)
    if config.backend is WalkingBackend.GEAR_SONIC:
        from rs_imle_policy.controllers.g1.locomotion.gear_sonic import (
            GearSonicVelocityInterface,
        )

        return GearSonicVelocityInterface(config.gear_sonic)
    raise ValueError(f"unsupported walking backend: {config.backend}")


__all__ = ["PlanarVelocity", "VelocityInterface", "create_velocity_interface"]
