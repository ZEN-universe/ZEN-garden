"""Retrofitting technology element."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zen_garden.elements.retrofitting_technology.retrofitting_technology import (
        RetrofittingTechnology,
    )


def __getattr__(name: str) -> Any:
    if name == "RetrofittingTechnology":
        from zen_garden.elements.retrofitting_technology.retrofitting_technology import (  # noqa: E501
            RetrofittingTechnology,
        )

        return RetrofittingTechnology
    raise AttributeError(name)


__all__ = ["RetrofittingTechnology"]
