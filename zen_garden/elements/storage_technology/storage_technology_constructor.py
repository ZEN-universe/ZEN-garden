"""Constructor for the StorageTechnology elements."""

import logging

import numpy as np
import xarray as xr
from typing_extensions import override

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.storage_technology import StorageTechnology
from zen_garden.elements.storage_technology.constraints import (
    STORAGE_TECHNOLOGY_CONSTRAINTS,
)

logger = logging.getLogger(__name__)


class StorageTechnologyConstructor(ModelConstructor):
    element_class = StorageTechnology
    constraints = STORAGE_TECHNOLOGY_CONSTRAINTS

    @override
    def construct_vars(self):
        logger.info("Constructing variables for StorageTechnology")

        def flow_storage_bounds(index_values, index_list):
            """Return bounds of carrier_flow for bigM expression.

            :param index_values: list of tuples with the index values
            :param index_list: The names of the indices
            :return bounds: bounds of carrier_flow
            """
            # get the arrays
            tech_arr, node_arr, time_arr = self.zen_model.sets.tuple_to_arr(
                index_values, index_list
            )
            # convert operationTimeStep to time_step_year:
            #   operationTimeStep -> base_time_step -> time_step_year
            time_step_year = xr.DataArray(
                [
                    self.time_steps.convert_time_step_operation2year(time)
                    for time in time_arr.data
                ]
            )
            lower = (
                self.zen_model.variables["capacity"]
                .lower.loc[tech_arr, "power", node_arr, time_step_year]
                .data
            )
            upper = (
                self.zen_model.variables["capacity"]
                .upper.loc[tech_arr, "power", node_arr, time_step_year]
                .data
            )
            return np.stack([lower, upper], axis=-1)

        for variable in self.variables:
            if (
                variable.name == "charge_storage_binary"
                and not self.config.system.storage_charge_discharge_binary
            ):
                continue

            if variable.name in ["flow_storage_charge", "flow_storage_discharge"]:
                # Exceptional bounds, masks or indices
                index_values, index_names = self.create_custom_set(
                    [
                        "set_storage_technologies",
                        "set_nodes",
                        "set_time_steps_operation",
                    ],
                )
                index_sets = index_values, index_names
                bounds = flow_storage_bounds(index_values, index_names)
            else:
                # Standard behavior
                index_sets = self.create_custom_set(variable.indices)
                bounds = variable.get_bounds()

            self.zen_model.add_variable(
                name=variable.name,
                index_sets=index_sets,
                binary=variable.binary,
                bounds=bounds,
                doc=variable.doc,
                unit_category=variable.unit_category,
            )
