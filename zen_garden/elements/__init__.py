from zen_garden.elements.carrier import Carrier
from zen_garden.elements.conversion_technology import ConversionTechnology
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.elements.retrofitting_technology import RetrofittingTechnology
from zen_garden.elements.storage_technology import StorageTechnology
from zen_garden.elements.technology import Technology
from zen_garden.elements.transport_technology import TransportTechnology
from zen_garden.model.element import Element

# The order matters because technologies populate ModelSchema.set_carriers,
# which is read before carrier elements are registered.
# for the carrier class
ELEMENT_TYPE_CLASSES: dict[str, type[Element]] = {
    "Technology": Technology,
    "ConversionTechnology": ConversionTechnology,
    "RetrofittingTechnology": RetrofittingTechnology,
    "StorageTechnology": StorageTechnology,
    "TransportTechnology": TransportTechnology,
    "Carrier": Carrier,
}

__all__ = [
    "Carrier",
    "ConversionTechnology",
    "Element",
    "EnergySystem",
    "RetrofittingTechnology",
    "StorageTechnology",
    "Technology",
    "TransportTechnology",
]
