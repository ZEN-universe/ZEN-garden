"""Constructor for the StorageTechnology elements."""

import logging

import numpy as np
import xarray as xr
from typing_extensions import override

from zen_garden.constraints.storage_technology import (
    STORAGE_TECHNOLOGY_CONSTRAINTS,
    ChargeDischargeBinaryConstraint,
    StorageTechnologyCapexConstraint,
)
from zen_garden.elements.element_constructor import ElementConstructor
from zen_garden.elements.storage_technology import StorageTechnology

logger = logging.getLogger(__name__)


class StorageTechnologyConstructor(ElementConstructor):
    element_class = StorageTechnology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.storage_technology.StorageTechnology`.

        :return: True if there are elements, False otherwise
        """
        return True

    @override
    def construct_sets(self):
        logger.info("Constructing sets for StorageTechnology")

    @override
    def construct_params(self):
        logger.info("Constructing parameters for StorageTechnology")
        # energy to power ratio
        self.add_parameter(
            name="energy_to_power_ratio_min",
            index_names=["set_storage_technologies"],
            doc="power to energy ratio for storage technologies - lower bound",
        )
        self.add_parameter(
            name="energy_to_power_ratio_max",
            index_names=["set_storage_technologies"],
            doc="power to energy ratio for storage technologies - upper bound",
        )
        # efficiency charge
        self.add_parameter(
            name="efficiency_charge",
            index_names=[
                "set_storage_technologies",
                "set_nodes",
                "set_years",
            ],
            doc="efficiency during charging for storage technologies",
        )
        # efficiency discharge
        self.add_parameter(
            name="efficiency_discharge",
            index_names=[
                "set_storage_technologies",
                "set_nodes",
                "set_years",
            ],
            doc="efficiency during discharging for storage technologies",
        )
        #  flow_storage_inflow
        self.add_parameter(
            name="flow_storage_inflow",
            index_names=[
                "set_storage_technologies",
                "set_nodes",
                "set_time_steps_operation",
            ],
            doc="energy inflow in storage technologies",
        )
        # self discharge
        self.add_parameter(
            name="self_discharge",
            index_names=["set_storage_technologies", "set_nodes"],
            doc="self discharge of storage technologies",
        )
        # capex specific
        self.add_parameter(
            name="capex_specific_storage",
            index_names=[
                "set_storage_technologies",
                "set_capacity_types",
                "set_nodes",
                "set_years",
            ],
            capacity_types=True,
            doc="specific capex of storage technologies",
        )

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
            index_sets=(index_values, index_names),
            bounds=bounds,
            doc="carrier flow into storage technology on node i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # flow of carrier on node out of storage
        self.zen_model.add_variable(
            name="flow_storage_discharge",
            index_sets=(index_values, index_names),
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
            index_sets=(index_values, index_names),
            bounds=(0, np.inf),
            doc="storage spillage of storage technology on node i in each "
            "storage time step",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # charge discharge binary
        if self.config.system.storage_charge_discharge_binary:
            self.zen_model.add_variable(
                name="charge_storage_binary",
                index_sets=(index_values, index_names),
                binary=True,
                doc="charge binary for storage technology",
                unit_category=None,
            )

    def construct_constraints(self):
        logger.info("Constructing constraints for StorageTechnology")

        for StorageTechnologyConstraint in STORAGE_TECHNOLOGY_CONSTRAINTS:
            StorageTechnologyConstraint(
                self.config, self.zen_model, self.energy_system, self.time_steps
            ).build()

        # Linear Capex
        index_values, index_names = self.create_custom_set(
            [
                "set_storage_technologies",
                "set_capacity_types",
                "set_nodes",
                "set_years",
            ],
        )
        StorageTechnologyCapexConstraint(
            self.config, self.zen_model, self.energy_system, self.time_steps
        ).build(index_values, index_names)

        # avoid simultaneous charge and discharge
        if self.config.system.storage_charge_discharge_binary:
            ChargeDischargeBinaryConstraint(
                self.config, self.zen_model, self.energy_system, self.time_steps
            ).build()
