"""storage technology parameters."""

from zen_garden.topology.generic_parameter import GenericParameter

from .capex_specific_storage import CapexSpecificStorage
from .efficiency_charge import EfficiencyCharge
from .efficiency_discharge import EfficiencyDischarge
from .energy_to_power_ratio_max import EnergyToPowerRatioMax
from .energy_to_power_ratio_min import EnergyToPowerRatioMin
from .flow_storage_inflow import FlowStorageInflow
from .self_discharge import SelfDischarge

STORAGE_TECHNOLOGY_PARAMETERS: list[type[GenericParameter]] = [
    EnergyToPowerRatioMin,
    EnergyToPowerRatioMax,
    EfficiencyCharge,
    EfficiencyDischarge,
    FlowStorageInflow,
    SelfDischarge,
    CapexSpecificStorage,
]

__all__ = [
    "EnergyToPowerRatioMin",
    "EnergyToPowerRatioMax",
    "EfficiencyCharge",
    "EfficiencyDischarge",
    "FlowStorageInflow",
    "SelfDischarge",
    "CapexSpecificStorage",
    "STORAGE_TECHNOLOGY_PARAMETERS",
]
