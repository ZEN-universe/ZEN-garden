from zen_garden.elements.carrier import Carrier
from zen_garden.elements.carrier_constructor import CarrierConstructor
from zen_garden.elements.conversion_technology import ConversionTechnology
from zen_garden.elements.conversion_technology_constructor import (
    ConversionTechnologyConstructor,
)
from zen_garden.elements.element import Element
from zen_garden.elements.element_constructor import ElementConstructor
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.elements.energy_system_constructor import EnergySystemConstructor
from zen_garden.elements.retrofitting_technology import RetrofittingTechnology
from zen_garden.elements.retrofitting_technology_constructor import (
    RetrofittingTechnologyConstructor,
)
from zen_garden.elements.storage_technology import StorageTechnology
from zen_garden.elements.storage_technology_constructor import (
    StorageTechnologyConstructor,
)
from zen_garden.elements.technology import Technology
from zen_garden.elements.technology_constructor import TechnologyConstructor
from zen_garden.elements.transport_technology import TransportTechnology
from zen_garden.elements.transport_technology_constructor import (
    TransportTechnologyConstructor,
)

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
    EnergySystemConstructor,
    CarrierConstructor,
    TechnologyConstructor,
    ConversionTechnologyConstructor,
    RetrofittingTechnologyConstructor,
    StorageTechnologyConstructor,
    TransportTechnologyConstructor,
]

__all__ = [
    "Carrier",
    "CarrierConstructor",
    "ConversionTechnology",
    "ConversionTechnologyConstructor",
    "Element",
    "ElementConstructor",
    "EnergySystem",
    "EnergySystemConstructor",
    "RetrofittingTechnology",
    "RetrofittingTechnologyConstructor",
    "StorageTechnology",
    "StorageTechnologyConstructor",
    "Technology",
    "TechnologyConstructor",
    "TransportTechnology",
    "TransportTechnologyConstructor",
]
