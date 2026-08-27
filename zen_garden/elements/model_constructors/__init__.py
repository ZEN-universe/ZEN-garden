from zen_garden.elements import ELEMENT_TYPE_CLASSES
from zen_garden.elements.carrier import Carrier
from zen_garden.elements.conversion_technology import ConversionTechnology
from zen_garden.elements.element import Element
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.retrofitting_technology import RetrofittingTechnology
from zen_garden.elements.storage_technology import StorageTechnology
from zen_garden.elements.technology import Technology
from zen_garden.elements.transport_technology import TransportTechnology

from ..energy_system.energy_system_constructor import EnergySystemConstructor
from ..technology.technology_constructor import TechnologyConstructor

# (constructor class, element class) pairs, in construction order. Most element
# types use the generic ModelConstructor; only types with genuine build behavior
# get a subclass.
MODEL_CONSTRUCTORS: list[tuple[type[ModelConstructor], type[Element]]] = [
    (EnergySystemConstructor, EnergySystem),
    (ModelConstructor, Carrier),
    (TechnologyConstructor, Technology),
    (ModelConstructor, ConversionTechnology),
    (ModelConstructor, RetrofittingTechnology),
    (ModelConstructor, StorageTechnology),
    (ModelConstructor, TransportTechnology),
]

# Guard against MODEL_CONSTRUCTORS drifting out of sync with the element classes:
# every element type must be built exactly once and vice versa.
_expected_element_classes: set[type] = {EnergySystem, *ELEMENT_TYPE_CLASSES.values()}
_actual_element_classes: list[type] = [pair[1] for pair in MODEL_CONSTRUCTORS]
assert len(_actual_element_classes) == len(set(_actual_element_classes)), (
    "MODEL_CONSTRUCTORS contains duplicate element classes"
)
assert set(_actual_element_classes) == _expected_element_classes, (
    "MODEL_CONSTRUCTORS is out of sync with the element classes: "
    f"missing {_expected_element_classes.difference(_actual_element_classes)}, "
    f"unexpected {set(_actual_element_classes).difference(_expected_element_classes)}"
)

__all__ = [
    "EnergySystemConstructor",
    "ModelConstructor",
    "TechnologyConstructor",
    "MODEL_CONSTRUCTORS",
]
