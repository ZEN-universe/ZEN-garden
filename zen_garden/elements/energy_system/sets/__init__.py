"""Energy-system set specifications."""

from zen_garden.topology.generic_set import GenericSet

from .set_carriers import SetCarriers
from .set_edges import SetEdges
from .set_elements import SetElements
from .set_hours_all_years import SetHoursAllYears
from .set_nodes import SetNodes
from .set_nodes_on_edges import SetNodesOnEdges
from .set_technologies import SetTechnologies
from .set_time_steps_operation import SetTimeStepsOperation
from .set_time_steps_storage import SetTimeStepsStorage
from .set_years import SetYears
from .set_years_entire_horizon import SetYearsEntireHorizon

ENERGY_SYSTEM_SETS: list[type[GenericSet]] = [
    SetNodes,
    SetEdges,
    SetNodesOnEdges,
    SetCarriers,
    SetTechnologies,
    SetElements,
    SetHoursAllYears,
    SetYears,
    SetYearsEntireHorizon,
    SetTimeStepsOperation,
    SetTimeStepsStorage,
]

__all__ = [
    "ENERGY_SYSTEM_SETS",
    "SetCarriers",
    "SetEdges",
    "SetElements",
    "SetHoursAllYears",
    "SetNodes",
    "SetNodesOnEdges",
    "SetTechnologies",
    "SetTimeStepsOperation",
    "SetTimeStepsStorage",
    "SetYears",
    "SetYearsEntireHorizon",
]
