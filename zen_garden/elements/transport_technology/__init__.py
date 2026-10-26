"""Transport technology element."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zen_garden.elements.transport_technology.transport_technology import (
        TransportTechnology,
    )


def __getattr__(name: str) -> Any:
    if name == "TransportTechnology":
        from zen_garden.elements.transport_technology.transport_technology import (
            TransportTechnology,
        )

        return TransportTechnology
    raise AttributeError(name)


__all__ = ["TransportTechnology"]
