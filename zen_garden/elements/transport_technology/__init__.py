"""Transport Technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.transport_technology.constraints.capacity_factor_transport_constraint import (
    CapacityFactorTransportConstraint,
)
from zen_garden.elements.transport_technology.constraints.opex_emissions_technology_transport_constraint import (
    OpexEmissionsTechnologyTransportConstraint,
)
from zen_garden.elements.transport_technology.constraints.transport_technology_capex_constraint import (
    TransportTechnologyCapexConstraint,
)
from zen_garden.elements.transport_technology.constraints.transport_technology_losses_flow_constraint import (
    TransportTechnologyLossesFlowConstraint,
)
from zen_garden.elements.transport_technology.transport_technology import (
    TransportTechnology,
)

TRANSPORT_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorTransportConstraint,
    OpexEmissionsTechnologyTransportConstraint,
    TransportTechnologyLossesFlowConstraint,
    TransportTechnologyCapexConstraint,
]

__all__ = [
    "TransportTechnology",
    "CapacityFactorTransportConstraint",
    "OpexEmissionsTechnologyTransportConstraint",
    "TransportTechnologyCapexConstraint",
    "TransportTechnologyLossesFlowConstraint",
]
