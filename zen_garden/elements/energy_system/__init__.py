"""Energy system element."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zen_garden.elements.energy_system.energy_system import EnergySystem


def __getattr__(name: str) -> Any:
    if name == "EnergySystem":
        from zen_garden.elements.energy_system.energy_system import EnergySystem

        return EnergySystem
    raise AttributeError(name)


__all__ = ["EnergySystem"]
