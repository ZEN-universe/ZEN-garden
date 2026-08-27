import numpy as np
import xarray as xr

from zen_garden.model.components.multi_index_helper import MultiIndexHelper
from zen_garden.topology.generic_variable import GenericVariable


class TechnologyInstallation(GenericVariable):
    """Variable for technology installation."""

    name = "technology_installation"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Binary variable indicating installation of technology"
    unit_category = {}
    binary = True

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None

    @classmethod
    def get_mask(cls, model_constructor, index_sets):
        zen_model = model_constructor.zen_model
        mask = xr.DataArray(
            False,
            coords=[
                zen_model.lp_model.variables.coords["set_years"],
                zen_model.lp_model.variables.coords["set_technologies"],
                zen_model.lp_model.variables.coords["set_capacity_types"],
                zen_model.lp_model.variables.coords["set_location"],
            ],
        )
        technologies = list(zen_model.sets["set_transport_technologies"])
        if technologies:
            edges = list(zen_model.sets["set_edges"])
            sub_mask = (
                zen_model.parameters.distance.loc[technologies, edges]
                * zen_model.parameters.capex_per_distance_transport.loc[
                    technologies, edges
                ]
                != 0
            ).rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
            mask.loc[:, technologies, :, edges] |= sub_mask

        mask |= (
            zen_model.parameters.capacity_addition_min.notnull()
            & (zen_model.parameters.capacity_addition_min != 0)
        )
        index_values, index_names = index_sets
        index = MultiIndexHelper(index_values, index_names)
        sub_mask = (
            zen_model.parameters.capacity_addition_max.notnull()
            & (zen_model.parameters.capacity_addition_max != np.inf)
            & (zen_model.parameters.capacity_addition_max != 0)
        )
        for tech, capacity_type in index.get_unique([0, 1]):
            locations = index.get_values(
                locs=[tech, capacity_type], levels=2, unique=True
            )
            mask.loc[:, tech, capacity_type, locations] |= sub_mask.loc[
                tech, capacity_type
            ]
        return mask
