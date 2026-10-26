"""Operational mask derived from available power capacity."""

from zen_garden.elements.energy_system.sets.set_representative_hours import (
    SetRepresentativeHours,
)
from zen_garden.elements.technology.masks.capacity_available_mask import (
    CapacityAvailableMask,
)
from zen_garden.model.component_types.mask import GenericMask


class PowerOperationMask(GenericMask):
    """Technology power capacity available in each operational time step."""

    name = "power_operation"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (CapacityAvailableMask,)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        capacity = registry[CapacityAvailableMask]
        representative_hours = list(
            model_constructor.optimization_model.sets[SetRepresentativeHours]
        )
        return capacity.expand_dims(
            {"set_representative_hours": representative_hours}, axis=-1
        )
