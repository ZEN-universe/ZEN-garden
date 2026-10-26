"""Mask for storage state variables and constraints."""

from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.storage_technology.masks.energy_capacity_available_mask import (  # noqa: E501
    EnergyCapacityAvailableMask,
)
from zen_garden.elements.storage_technology.masks.storage_inflow_mask import (
    StorageInflowMask,
)
from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.mask import GenericMask


class StorageStateMask(GenericMask):
    """Storage states required for a year containing capacity or inflow."""

    name = "storage_state"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours_storage",
    )
    dependencies = (
        CapacityAvailableMask,
        EnergyCapacityAvailableMask,
        StorageInflowMask,
    )

    @classmethod
    def get_mask(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        power_available = registry[CapacityAvailableMask]
        energy_available = registry[EnergyCapacityAvailableMask]
        inflow = registry[StorageInflowMask]
        storage_technologies = list(model.sets[SetStorageTechnologies])
        nodes = list(model.sets[SetNodes])
        index_values, index_names = model_constructor.create_custom_set(
            [
                "set_storage_technologies",
                "set_nodes",
                "set_years",
                "set_representative_hours_storage",
            ]
        )
        valid_storage_hours = model.sets.indices_to_mask(
            index_values, index_names, (0, 0)
        )[0].rename(
            {
                "set_storage_technologies": "set_technologies",
                "set_nodes": "set_location",
            }
        )
        if not storage_technologies:
            return valid_storage_hours
        year_active = power_available.sel(
            set_technologies=storage_technologies,
            set_location=nodes,
        ) | energy_available.sel(
            set_technologies=storage_technologies,
            set_location=nodes,
        )
        year_active |= inflow.any("set_representative_hours")
        return valid_storage_hours & year_active
