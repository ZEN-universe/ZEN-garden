"""Transport technology constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

from .capacity_factor_transport_constraint import (
    CapacityFactorTransportConstraint,
)

TRANSPORT_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorTransportConstraint,
]

__all__ = [
    "CapacityFactorTransportConstraint",
]
