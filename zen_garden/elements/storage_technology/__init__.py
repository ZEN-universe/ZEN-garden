"""Storage technology element."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zen_garden.elements.storage_technology.storage_technology import (
        StorageTechnology,
    )


def __getattr__(name: str) -> Any:
    if name == "StorageTechnology":
        from zen_garden.elements.storage_technology.storage_technology import (
            StorageTechnology,
        )

        return StorageTechnology
    raise AttributeError(name)


__all__ = ["StorageTechnology"]
