"""Transport Technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint

from .capacity_factor_transport_constraint import CapacityFactorTransportConstraint
from .opex_emissions_technology_transport_constraint import (
    OpexEmissionsTechnologyTransportConstraint,
)
from .transport_technology_capex_constraint import TransportTechnologyCapexConstraint
from .transport_technology_losses_flow_constraint import (
    TransportTechnologyLossesFlowConstraint,
)

TRANSPORT_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorTransportConstraint,
    OpexEmissionsTechnologyTransportConstraint,
    TransportTechnologyLossesFlowConstraint,
]

__all__ = [
    "CapacityFactorTransportConstraint",
    "OpexEmissionsTechnologyTransportConstraint",
    "TransportTechnologyCapexConstraint",
    "TransportTechnologyLossesFlowConstraint",
]
