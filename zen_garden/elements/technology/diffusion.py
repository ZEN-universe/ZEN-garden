"""Shared technology-diffusion calculations."""

import pandas as pd

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.parameters.max_diffusion_rate import (
    MaxDiffusionRate,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition


def diffusion_rate(model_constructor):
    """Return maximum capacity growth over one investment interval."""
    model = model_constructor.optimization_model
    interval = model_constructor.optimization_model.parameters[IntervalBetweenYears]
    rate = (1 + model.parameters[MaxDiffusionRate]) ** interval - 1
    return rate.broadcast_like(model.variables[CapacityAddition].lower)


def storage_energy_diffusion_rate(model_constructor):
    """Return storage energy-capacity growth over one investment interval."""
    model = model_constructor.optimization_model
    interval = model.parameters[IntervalBetweenYears]
    technologies = model.sets[SetStorageTechnologies]
    rate = (1 + model.parameters[MaxDiffusionRate]) ** interval - 1
    rate = rate.sel(set_technologies=technologies)
    rate = rate.rename(
        {
            dimension: replacement
            for dimension, replacement in {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }.items()
            if dimension in rate.dims
        }
    )
    return rate.broadcast_like(model.variables[CapacityAdditionEnergy].lower)


def technology_type_mask(model_constructor):
    """Return one for conversion/storage technologies and zero for transport."""
    model = model_constructor.optimization_model
    mask = pd.Series(index=pd.Index(model.sets[SetTechnologies]), data=1)
    mask.index.name = "set_technologies"
    mask[mask.index.isin(model.sets[SetTransportTechnologies])] = 0
    return mask.to_xarray()


def technology_location_mask(model_constructor):
    """Select valid node technologies and valid edge transport technologies."""
    model = model_constructor.optimization_model
    capacity_addition = model.variables[CapacityAddition]
    type_mask = technology_type_mask(model_constructor)
    location_mask = pd.Series(
        index=pd.Index(capacity_addition.coords["set_location"]), data=1
    )
    location_mask.index.name = "set_location"
    location_mask[location_mask.index.isin(model.sets[SetEdges])] = 0
    location_mask = location_mask.to_xarray()
    return ((1 - type_mask) & (1 - location_mask)) | (type_mask & location_mask)
