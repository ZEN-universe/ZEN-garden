"""Shared technology-diffusion calculations."""

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.parameters.max_diffusion_rate import (
    MaxDiffusionRate,
)
from zen_garden.elements.technology.sets.relation_technology_location import (
    RelationTechnologyLocation,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
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
    location_mask = model.sets.relation_mask(RelationTechnologyLocation)
    return location_mask.sel(set_location=model.sets[SetNodes].members).any(
        "set_location"
    )


def technology_location_mask(model_constructor):
    """Select valid node technologies and valid edge transport technologies."""
    model = model_constructor.optimization_model
    return model.sets.relation_mask(RelationTechnologyLocation)
