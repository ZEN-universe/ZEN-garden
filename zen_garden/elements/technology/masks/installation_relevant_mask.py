"""Mask for installation binaries with a nonzero model role."""

import numpy as np
import xarray as xr

from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.elements.technology.parameters.capacity_addition_max import (
    CapacityAdditionMax,
)
from zen_garden.elements.technology.parameters.capacity_addition_min import (
    CapacityAdditionMin,
)
from zen_garden.elements.technology.parameters.capacity_limit import CapacityLimit
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.transport_technology.parameters.capex_per_distance_transport import (  # noqa: E501
    CapexPerDistanceTransport,
)
from zen_garden.elements.transport_technology.parameters.distance import Distance
from zen_garden.model.component_types.mask import GenericMask


class InstallationRelevantMask(GenericMask):
    """Positions where the installation binary has a nonzero model role."""

    name = "installation_relevant"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_mask(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        mask = xr.zeros_like(model.parameters[CapacityLimit], dtype=bool)
        minimum = model.parameters[CapacityAdditionMin]
        maximum = model.parameters[CapacityAdditionMax]
        mask |= minimum.notnull() & (minimum != 0)
        mask |= maximum.notnull() & np.isfinite(maximum) & (maximum != 0)

        technologies = list(model.sets[SetTransportTechnologies])
        if technologies:
            edges = list(model.sets[SetEdges])
            distance_cost = (
                model.parameters[Distance].loc[technologies, edges]
                * model.parameters[CapexPerDistanceTransport].loc[technologies, edges]
                != 0
            ).rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
            mask.loc[technologies, edges, :] |= distance_cost
        return mask
