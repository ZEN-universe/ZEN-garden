"""Mask for transport routes with finite distance."""

import numpy as np
import xarray as xr

from zen_garden.model.component_types.mask import GenericMask


class FiniteTransportRouteMask(GenericMask):
    """Transport routes with finite distance."""

    name = "finite_transport_route"

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        technologies = list(model.sets["set_transport_technologies"])
        edges = list(model.sets["set_edges"])
        mask = xr.DataArray(
            False,
            coords={"set_technologies": technologies, "set_location": edges},
            dims=("set_technologies", "set_location"),
        )
        if technologies:
            mask.loc[technologies, edges] = np.isfinite(
                model.parameters.distance.loc[technologies, edges]
            ).data
        return mask
