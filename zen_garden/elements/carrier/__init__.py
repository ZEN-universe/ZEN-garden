"""Carrier element."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zen_garden.elements.carrier.carrier import Carrier


def __getattr__(name: str) -> Any:
    if name == "Carrier":
        from zen_garden.elements.carrier.carrier import Carrier

        return Carrier
    raise AttributeError(name)


__all__ = ["Carrier"]
