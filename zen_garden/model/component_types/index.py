"""Metadata for structural indices that are not registered model sets."""

from typing import ClassVar


class GenericIndex:
    """Describe a structural model dimension used to construct custom sets."""

    name: ClassVar[str]
    symbol: ClassVar[str]
    doc: ClassVar[str]
    index_set: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attribute in ("name", "symbol", "doc"):
            value = getattr(cls, attribute, None)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"{cls.__name__}.{attribute} must be a non-empty string"
                )


class SetLocations(GenericIndex):
    """Technology locations selected from nodes or edges."""

    name = "set_location"
    symbol = r"p \in \mathcal{P}"
    doc = "Locations where technologies can be installed."


class SetConstructionTimeSteps(GenericIndex):
    """Investment years used to represent construction time."""

    name = "set_time_steps_construction"
    symbol = r"v \in \mathcal{Y}^{\mathrm{construction}}"
    doc = "Investment years used to account for technology construction time."
    index_set = "set_technologies"


class SetOperationTimeSteps(GenericIndex):
    """Operational time steps represented by modeled years and hours."""

    name = "set_time_steps_operation"
    symbol = r"t \in \mathcal{T}_{y}"
    doc = "Operational time steps used in the optimization."


class SetStorageTimeSteps(GenericIndex):
    """Time steps used to track storage levels."""

    name = "set_time_steps_storage"
    symbol = r"\tilde{t} \in \widetilde{\mathcal{T}}_{y}"
    doc = "Time steps used to track storage levels."


STRUCTURAL_INDICES: tuple[type[GenericIndex], ...] = (
    SetLocations,
    SetConstructionTimeSteps,
    SetOperationTimeSteps,
    SetStorageTimeSteps,
)
