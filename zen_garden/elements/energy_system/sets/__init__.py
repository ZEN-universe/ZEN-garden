"""Energy-system set specifications."""

from zen_garden.model.component_types.set import SetDeclaration

from .relation_edge_node import RelationEdgeEndNode, RelationEdgeStartNode
from .relation_year_representative_hour_storage import (
    RelationYearRepresentativeHourStorage,
)
from .set_capacity_types import SetCapacityTypes
from .set_carriers import SetCarriers
from .set_edges import SetEdges
from .set_elements import SetElements
from .set_hours import SetHours
from .set_location import SetLocation
from .set_nodes import SetNodes
from .set_representative_hours import SetRepresentativeHours
from .set_representative_hours_storage import SetRepresentativeHoursStorage
from .set_technologies import SetTechnologies
from .set_years import SetYears
from .set_years_entire_horizon import SetYearsEntireHorizon

ENERGY_SYSTEM_SETS: list[SetDeclaration] = [
    SetLocation,
    SetNodes,
    SetEdges,
    SetCarriers,
    SetCapacityTypes,
    SetTechnologies,
    SetElements,
    SetHours,
    SetYears,
    SetYearsEntireHorizon,
    SetRepresentativeHours,
    SetRepresentativeHoursStorage,
    RelationYearRepresentativeHourStorage,
    RelationEdgeStartNode,
    RelationEdgeEndNode,
]

__all__ = [
    "ENERGY_SYSTEM_SETS",
    "SetCarriers",
    "SetCapacityTypes",
    "SetEdges",
    "SetElements",
    "SetHours",
    "SetLocation",
    "SetNodes",
    "SetTechnologies",
    "SetRepresentativeHours",
    "SetRepresentativeHoursStorage",
    "SetYears",
    "SetYearsEntireHorizon",
    "RelationEdgeEndNode",
    "RelationEdgeStartNode",
    "RelationYearRepresentativeHourStorage",
]
