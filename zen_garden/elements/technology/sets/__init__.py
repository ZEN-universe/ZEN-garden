"""Technology set specifications."""

from zen_garden.model.component_types.set import GenericSet

from .set_conversion_technologies import SetConversionTechnologies
from .set_reference_carriers import SetReferenceCarriers
from .set_retrofitting_technologies import SetRetrofittingTechnologies
from .set_storage_technologies import SetStorageTechnologies
from .set_technologies_existing import SetTechnologiesExisting
from .set_transport_technologies import SetTransportTechnologies

TECHNOLOGY_SETS: list[type[GenericSet]] = [
    SetConversionTechnologies,
    SetRetrofittingTechnologies,
    SetTransportTechnologies,
    SetStorageTechnologies,
    SetTechnologiesExisting,
    SetReferenceCarriers,
]
__all__ = [
    "TECHNOLOGY_SETS",
    "SetConversionTechnologies",
    "SetReferenceCarriers",
    "SetRetrofittingTechnologies",
    "SetStorageTechnologies",
    "SetTechnologiesExisting",
    "SetTransportTechnologies",
]
