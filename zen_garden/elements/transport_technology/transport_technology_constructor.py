"""Constructor for the TransportTechnology elements."""

import logging

import numpy as np
import xarray as xr
from typing_extensions import override

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.transport_technology import TransportTechnology
from zen_garden.elements.transport_technology.constraints import (
    TRANSPORT_TECHNOLOGY_CONSTRAINTS,
)

logger = logging.getLogger(__name__)


class TransportTechnologyConstructor(ModelConstructor):
    element_class = TransportTechnology
    constraints = TRANSPORT_TECHNOLOGY_CONSTRAINTS
    parameters = TransportTechnology.own_parameters
    variables = TransportTechnology.variables

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.transport_technology.TransportTechnology`.

        :return: True if there are elements, False otherwise
        """
        return True

    @override
    def construct_sets(self):
        logger.info("Constructing sets for TransportTechnology")

    @override
    def construct_vars(self):
        logger.info("Constructing variables for TransportTechnology")

        def flow_transport_bounds(index_values, index_list):
            """Return bounds of carrier_flow for bigM expression.

            :param index_values: list of tuples with the index values
            :param index_list: The names of the indices
            :return bounds: bounds of carrier_flow
            """
            # get the arrays
            tech_arr, edge_arr, time_arr = self.zen_model.sets.tuple_to_arr(
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
                .lower.loc[tech_arr, "power", edge_arr, time_step_year]
                .data
            )
            upper = (
                self.zen_model.variables["capacity"]
                .upper.loc[tech_arr, "power", edge_arr, time_step_year]
                .data
            )
            return np.stack([lower, upper], axis=-1)

        # # flow of carrier on edge
        # index_values, index_names = self.create_custom_set(
        #     ["set_transport_technologies", "set_edges", "set_time_steps_operation"]
        # )
        # bounds = flow_transport_bounds(index_values, index_names)
        # self.zen_model.add_variable(
        #     name="flow_transport",
        #     index_sets=(index_values, index_names),
        #     bounds=bounds,
        #     doc="carrier flow through transport technology on edge i and time t",
        #     unit_category={"energy_quantity": 1, "time": -1},
        # )
        # loss of carrier on edge
        # self.zen_model.add_variable(
        #     name="flow_transport_loss",
        #     index_sets=(index_values, index_names),
        #     bounds=(0, np.inf),
        #     doc="carrier flow lost due to resistances etc. by transporting carrier "
        #     "through transport technology on edge i and time t",
        #     unit_category={"energy_quantity": 1, "time": -1},
        # )

        for variable in self.variables:
            if variable.name in ["flow_transport"]:
                # Exceptional bounds, masks or indices
                index_values, index_names = self.create_custom_set(variable.indices)
                index_sets = index_values, index_names
                bounds = flow_transport_bounds(index_values, index_names)
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

    @override
    def construct_expressions(self):
        """Construct reusable transport coefficients."""
        parameters = self.zen_model.parameters
        transport_technologies = self.zen_model.sets["set_transport_technologies"]

        self.zen_model.add_expression(
            "transport_capex_distance",
            parameters.distance * parameters.capex_per_distance_transport,
        )
        self.zen_model.add_expression(
            "transport_loss_factor_effective",
            parameters.transport_loss_factor,
        )
        self.zen_model.add_expression(
            "transport_carbon_intensity_effective",
            parameters.carbon_intensity_technology.sel(
                set_technologies=transport_technologies
            ),
        )
