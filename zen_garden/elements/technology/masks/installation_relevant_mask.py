"""Mask for installation binaries with a nonzero model role."""

import numpy as np
import xarray as xr

from zen_garden.model.component_types.mask import GenericMask


class InstallationRelevantMask(GenericMask):
    """Positions where the installation binary has a nonzero model role."""

    name = "installation_relevant"

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        mask = xr.zeros_like(model.parameters.capacity_limit, dtype=bool)
        minimum = model.parameters.capacity_addition_min
        maximum = model.parameters.capacity_addition_max
        mask |= minimum.notnull() & (minimum != 0)
        mask |= maximum.notnull() & np.isfinite(maximum) & (maximum != 0)

        technologies = list(model.sets["set_transport_technologies"])
        if technologies:
            edges = list(model.sets["set_edges"])
            distance_cost = (
                model.parameters.distance.loc[technologies, edges]
                * model.parameters.capex_per_distance_transport.loc[technologies, edges]
                != 0
            ).rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
            mask.loc[technologies, :, edges, :] |= distance_cost
        return mask
