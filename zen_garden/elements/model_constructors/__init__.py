from zen_garden.elements import ELEMENT_TYPE_CLASSES
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.elements.model_constructor import ModelConstructor

from ..carrier.carrier_constructor import CarrierConstructor
from ..conversion_technology.conversion_technology_constructor import (
    ConversionTechnologyConstructor,
)
from ..energy_system.energy_system_constructor import (
    EnergySystemConstructor,
)
from ..retrofitting_technology.retrofitting_technology_constructor import (
    RetrofittingTechnologyConstructor,
)
from ..storage_technology.storage_technology_constructor import (
    StorageTechnologyConstructor,
)
from ..technology.technology_constructor import TechnologyConstructor
from ..transport_technology.transport_technology_constructor import (
    TransportTechnologyConstructor,
)

# Order matters: it is the order in which sets, parameters, variables and
# constraints are constructed.
MODEL_CONSTRUCTORS: list[type[ModelConstructor]] = [
    EnergySystemConstructor,
    CarrierConstructor,
    TechnologyConstructor,
    ConversionTechnologyConstructor,
    RetrofittingTechnologyConstructor,
    StorageTechnologyConstructor,
    TransportTechnologyConstructor,
]

# Guard against MODEL_CONSTRUCTORS drifting out of sync with the element classes:
# every element type must have exactly one constructor and vice versa.
_expected_element_classes: set[type] = {EnergySystem, *ELEMENT_TYPE_CLASSES.values()}
_actual_element_classes: list[type] = [c.element_class for c in MODEL_CONSTRUCTORS]
assert len(_actual_element_classes) == len(set(_actual_element_classes)), (
    "MODEL_CONSTRUCTORS contains duplicate element classes"
)
assert set(_actual_element_classes) == _expected_element_classes, (
    "MODEL_CONSTRUCTORS is out of sync with the element classes: "
    f"missing {_expected_element_classes.difference(_actual_element_classes)}, "
    f"unexpected {set(_actual_element_classes).difference(_expected_element_classes)}"
)

__all__ = [
    "CarrierConstructor",
    "EnergySystemConstructor",
    "ConversionTechnologyConstructor",
    "ModelConstructor",
    "RetrofittingTechnologyConstructor",
    "StorageTechnologyConstructor",
    "TechnologyConstructor",
    "TransportTechnologyConstructor",
]
