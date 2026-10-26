"""Mask for nonzero exogenous storage inflow."""

import xarray as xr

from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.energy_system.sets.set_representative_hours import (
    SetRepresentativeHours,
)
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.parameters.flow_storage_inflow import (
    FlowStorageInflow,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.mask import GenericMask


class StorageInflowMask(GenericMask):
    """Operational positions with nonzero exogenous storage inflow."""

    name = "storage_inflow"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_mask(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        technologies = list(model.sets[SetStorageTechnologies])
        if not technologies:
            return xr.DataArray(
                False,
                coords={
                    "set_technologies": technologies,
                    "set_location": list(model.sets[SetNodes]),
                    "set_years": list(model.sets[SetYears]),
                    "set_representative_hours": list(
                        model.sets[SetRepresentativeHours]
                    ),
                },
                dims=(
                    "set_technologies",
                    "set_location",
                    "set_years",
                    "set_representative_hours",
                ),
            )
        inflow = model.parameters[FlowStorageInflow]
        return (inflow.fillna(0) != 0).rename(
            {
                "set_storage_technologies": "set_technologies",
                "set_nodes": "set_location",
            }
        )
