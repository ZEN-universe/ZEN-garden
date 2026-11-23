"""Technology set specifications."""

from zen_garden.model.component_types.set import SetDeclaration

from .relation_technology_capacity_type import RelationTechnologyCapacityType
from .relation_technology_existing_capacity import RelationTechnologyExistingCapacity
from .relation_technology_location import RelationTechnologyLocation
from .relation_technology_reference_carrier import (
    RelationTechnologyReferenceCarrier,
)
from .set_conversion_technologies import SetConversionTechnologies
from .set_existing_capacity_ids import SetExistingCapacityIds
from .set_retrofitting_technologies import SetRetrofittingTechnologies
from .set_storage_technologies import SetStorageTechnologies
from .set_transport_technologies import SetTransportTechnologies

TECHNOLOGY_SETS: list[SetDeclaration] = [
    SetConversionTechnologies,
    SetRetrofittingTechnologies,
    SetTransportTechnologies,
    SetStorageTechnologies,
    SetExistingCapacityIds,
    RelationTechnologyExistingCapacity,
    RelationTechnologyReferenceCarrier,
    RelationTechnologyLocation,
    RelationTechnologyCapacityType,
]
__all__ = [
    "TECHNOLOGY_SETS",
    "RelationTechnologyCapacityType",
    "RelationTechnologyExistingCapacity",
    "RelationTechnologyLocation",
    "RelationTechnologyReferenceCarrier",
    "SetConversionTechnologies",
    "SetExistingCapacityIds",
    "SetRetrofittingTechnologies",
    "SetStorageTechnologies",
    "SetTransportTechnologies",
]
