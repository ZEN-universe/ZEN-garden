"""Transport technology constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

from .capacity_factor_transport_constraint import (
    CapacityFactorTransportConstraint,
)
from .carbon_emissions_technology_transport_constraint import (
    CarbonEmissionsTechnologyTransportConstraint,
)
from .opex_technology_transport_constraint import OpexTechnologyTransportConstraint
from .transport_technology_capex_constraint import (
    TransportTechnologyCapexConstraint,
)
from .transport_technology_losses_flow_constraint import (
    TransportTechnologyLossesFlowConstraint,
)

TRANSPORT_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorTransportConstraint,
    OpexTechnologyTransportConstraint,
    CarbonEmissionsTechnologyTransportConstraint,
    TransportTechnologyLossesFlowConstraint,
    TransportTechnologyCapexConstraint,
]

__all__ = [
    "CapacityFactorTransportConstraint",
    "CarbonEmissionsTechnologyTransportConstraint",
    "OpexTechnologyTransportConstraint",
    "TransportTechnologyCapexConstraint",
    "TransportTechnologyLossesFlowConstraint",
]
