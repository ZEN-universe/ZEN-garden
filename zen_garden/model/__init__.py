"""Explicit model package imports."""

from zen_garden.model.carrier import Carrier
from zen_garden.model.carrier.carrier import CarrierConstructor
from zen_garden.model.components.constraint import Constraint
from zen_garden.model.components.index_set import IndexSet
from zen_garden.model.components.parameter import DictParameter, Parameter
from zen_garden.model.components.variable import Variable
from zen_garden.model.components.zen_index import ZenIndex
from zen_garden.model.components.zen_set import ZenSet
from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.element import Element, ElementConstructor
from zen_garden.model.energy_system import EnergySystem
from zen_garden.model.generic_rule import GenericRule
from zen_garden.model.technology import (
    ConversionTechnology,
    RetrofittingTechnology,
    StorageTechnology,
    Technology,
    TransportTechnology,
)
from zen_garden.model.technology.conversion_technology import (
    ConversionTechnologyConstructor,
)
from zen_garden.model.technology.retrofitting_technology import (
    RetrofittingTechnologyConstructor,
)
from zen_garden.model.technology.storage_technology import StorageTechnologyConstructor
from zen_garden.model.technology.technology import TechnologyConstructor
from zen_garden.model.technology.transport_technology import (
    TransportTechnologyConstructor,
)
from zen_garden.model.time_steps import TimeStepsDicts
from zen_garden.model.zen_model import ZenModel

__all__ = [
    "Carrier",
    "Config",
    "Constraint",
    "Context",
    "ConversionTechnology",
    "DictParameter",
    "Element",
    "EnergySystem",
    "GenericRule",
    "IndexSet",
    "Parameter",
    "RetrofittingTechnology",
    "StorageTechnology",
    "Technology",
    "TimeStepsDicts",
    "TransportTechnology",
    "Variable",
    "ZenIndex",
    "ZenSet",
    "ZenModel",
]

# The order matters because ConversionTechnology calls
# EnergySystem.set_technology_of_carrier, which in turn modifies
# EnergySystem.set_carrier, which is read in ElementRegistry.add_elements
# for the carrier class
ELEMENT_TYPE_CLASSES: dict[str, type[Element]] = {
    "Technology": Technology,
    "ConversionTechnology": ConversionTechnology,
    "RetrofittingTechnology": RetrofittingTechnology,
    "StorageTechnology": StorageTechnology,
    "TransportTechnology": TransportTechnology,
    "Carrier": Carrier,
}

ELEMENT_CONSTRUCTORS: list[type[ElementConstructor]] = [
    CarrierConstructor,
    TechnologyConstructor,
    ConversionTechnologyConstructor,
    RetrofittingTechnologyConstructor,
    StorageTechnologyConstructor,
    TransportTechnologyConstructor,
]
