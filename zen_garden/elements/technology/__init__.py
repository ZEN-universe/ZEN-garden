"""Technology element."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zen_garden.elements.technology.technology import Technology


def __getattr__(name: str) -> Any:
    if name == "Technology":
        from zen_garden.elements.technology.technology import Technology

        return Technology
    raise AttributeError(name)


__all__ = ["Technology"]
