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
    parameters = StorageTechnology.own_parameters
    variables = StorageTechnology.variables

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.storage_technology.StorageTechnology`.

        :return: True if there are elements, False otherwise
        """
        return True

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

        # flow of carrier on node into storage
        index_values, index_names = self.create_custom_set(
            ["set_storage_technologies", "set_nodes", "set_time_steps_operation"],
        )
        bounds = flow_storage_bounds(index_values, index_names)
        self.zen_model.add_variable(
            name="flow_storage_charge",
            index_sets=self.create_custom_set(
            ["set_storage_technologies", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=bounds,
            doc="carrier flow into storage technology on node i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # flow of carrier on node out of storage
        self.zen_model.add_variable(
            name="flow_storage_discharge",
            index_sets=self.create_custom_set(
            ["set_storage_technologies", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=bounds,
            doc="carrier flow out of storage technology on node i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # storage level
        self.zen_model.add_variable(
            name="storage_level",
            index_sets=self.create_custom_set(
                ["set_storage_technologies", "set_nodes", "set_time_steps_storage"],
            ),
            bounds=(0, np.inf),
            doc="storage level of storage technology ón node in each storage time step",
            unit_category={"energy_quantity": 1},
        )
        # energy spillage
        self.zen_model.add_variable(
            name="flow_storage_spillage",
            index_sets=self.create_custom_set(
            ["set_storage_technologies", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0, np.inf),
            doc="storage spillage of storage technology on node i in each "
            "storage time step",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # charge discharge binary
        if self.config.system.storage_charge_discharge_binary:
            self.zen_model.add_variable(
                name="charge_storage_binary",
                index_sets=self.create_custom_set(
                ["set_storage_technologies", "set_nodes", "set_time_steps_operation"],
                ),
                binary=True,
                doc="charge binary for storage technology",
                unit_category=None,
            )
