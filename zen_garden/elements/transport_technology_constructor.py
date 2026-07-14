import logging
from typing import override

import numpy as np
import xarray as xr

from zen_garden.elements.element_constructor import ElementConstructor
from zen_garden.elements.transport_technology import TransportTechnology
from zen_garden.elements.transport_technology_rules import TransportTechnologyRules

logger = logging.getLogger(__name__)


class TransportTechnologyConstructor(ElementConstructor):
    element_class = TransportTechnology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class <Carrier>.

        :return: True if there are elements, False otherwise
        """
        return True

    @override
    def construct_sets(self):
        logger.info("Constructing sets for TransportTechnology")

    @override
    def construct_params(self):
        logger.info("Constructing parameters for TransportTechnology")

        # distance between nodes
        self.add_parameter(
            name="distance",
            index_names=["set_transport_technologies", "set_edges"],
            doc="distance between two nodes for transport technologies",
        )
        # capital cost per unit
        self.add_parameter(
            name="capex_specific_transport",
            index_names=[
                "set_transport_technologies",
                "set_edges",
                "set_years",
            ],
            doc="capex per unit for transport technologies",
        )
        # capital cost per distance
        self.add_parameter(
            name="capex_per_distance_transport",
            index_names=[
                "set_transport_technologies",
                "set_edges",
                "set_years",
            ],
            doc="capex per distance for transport technologies",
        )
        # carrier losses
        self.add_parameter(
            name="transport_loss_factor",
            index_names=["set_transport_technologies", "set_edges"],
            doc="linear carrier losses due to transport with transport technologies",
        )

    @override
    def construct_vars(self):
        logger.info("Constructing variables for TransportTechnology")

        model = self.zen_model.lp_model
        variables = self.zen_model.variables
        sets = self.zen_model.sets

        def flow_transport_bounds(index_values, index_list):
            """Return bounds of carrier_flow for bigM expression.

            :param index_values: list of tuples with the index values
            :param index_list: The names of the indices
            :return bounds: bounds of carrier_flow
            """
            # get the arrays
            tech_arr, edge_arr, time_arr = sets.tuple_to_arr(index_values, index_list)
            # convert operationTimeStep to time_step_year:
            #   operationTimeStep -> base_time_step -> time_step_year
            time_step_year = xr.DataArray(
                [
                    self.time_steps.convert_time_step_operation2year(time)
                    for time in time_arr.data
                ]
            )

            lower = (
                model.variables["capacity"]
                .lower.loc[tech_arr, "power", edge_arr, time_step_year]
                .data
            )
            upper = (
                model.variables["capacity"]
                .upper.loc[tech_arr, "power", edge_arr, time_step_year]
                .data
            )
            return np.stack([lower, upper], axis=-1)

        # flow of carrier on edge
        index_values, index_names = self.create_custom_set(
            ["set_transport_technologies", "set_edges", "set_time_steps_operation"]
        )
        bounds = flow_transport_bounds(index_values, index_names)
        variables.add_variable(
            name="flow_transport",
            index_sets=(index_values, index_names),
            bounds=bounds,
            doc="carrier flow through transport technology on edge i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # loss of carrier on edge
        variables.add_variable(
            name="flow_transport_loss",
            index_sets=(index_values, index_names),
            bounds=(0, np.inf),
            doc="carrier flow lost due to resistances etc. by transporting carrier "
            "through transport technology on edge i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for TransportTechnology")

        rules = TransportTechnologyRules(
            self.config, self.zen_model, self.energy_system, self.time_steps
        )

        # limit flow by capacity and max load
        rules.constraint_capacity_factor_transport()

        # opex and emissions constraint for transport technologies
        rules.constraint_opex_emissions_technology_transport()

        # carrier flow Losses
        rules.constraint_transport_technology_losses_flow()

        # capex of transport technologies
        index_values, index_list = self.create_custom_set(
            ["set_transport_technologies", "set_edges", "set_years"]
        )
        rules.constraint_transport_technology_capex(index_values, index_list)
