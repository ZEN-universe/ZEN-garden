"""Conversion technology element."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zen_garden.elements.conversion_technology.conversion_technology import (
        ConversionTechnology,
    )


def __getattr__(name: str) -> Any:
    if name == "ConversionTechnology":
        from zen_garden.elements.conversion_technology.conversion_technology import (
            ConversionTechnology,
        )

        return ConversionTechnology
    raise AttributeError(name)


__all__ = ["ConversionTechnology"]
