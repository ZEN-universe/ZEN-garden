"""Constructor for the ConversionTechnology elements."""

import logging

import numpy as np
import xarray as xr
from typing_extensions import override

from zen_garden.elements.conversion_technology import (
    ConversionTechnology,
)
from zen_garden.elements.conversion_technology.constraints import (
    CONVERSION_TECHNOLOGY_CONSTRAINTS,
    LinearCapexConstraint,
)
from zen_garden.elements.model_constructor import ModelConstructor

logger = logging.getLogger(__name__)


class ConversionTechnologyConstructor(ModelConstructor):
    element_class = ConversionTechnology
    parameters = ConversionTechnology.own_parameters

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.conversion_technology.ConversionTechnology`.

        :return: True if there are elements, False otherwise
        """
        return True

    @override
    def construct_sets(self):
        logger.info("Constructing sets for ConversionTechnology")
        # get input carriers
        input_carriers = self.element_registry.get_attribute_of_all_elements(
            self.element_class, "input_carrier"
        )
        output_carriers = self.element_registry.get_attribute_of_all_elements(
            self.element_class, "output_carrier"
        )
        reference_carrier = self.element_registry.get_attribute_of_all_elements(
            self.element_class, "reference_carrier"
        )
        dependent_carriers = {}
        for tech in input_carriers:
            dependent_carriers[tech] = input_carriers[tech] + output_carriers[tech]
            dependent_carriers[tech].remove(reference_carrier[tech][0])
        # input carriers of technology
        self.zen_model.add_set(
            name="set_input_carriers",
            data=input_carriers,
            doc="set of carriers that are an input to a specific conversion "
            "technology. Indexed by set_conversion_technologies",
            index_set="set_conversion_technologies",
        )
        # output carriers of technology
        self.zen_model.add_set(
            name="set_output_carriers",
            data=output_carriers,
            doc="set of carriers that are an output to a specific conversion "
            "technology. Indexed by set_conversion_technologies",
            index_set="set_conversion_technologies",
        )
        # dependent carriers of technology
        self.zen_model.add_set(
            name="set_dependent_carriers",
            data=dependent_carriers,
            doc="set of carriers that are an output to a specific conversion "
            "technology. Indexed by set_conversion_technologies",
            index_set="set_conversion_technologies",
        )

    @override
    def construct_vars(self):
        logger.info("Constructing variables for ConversionTechnology")

        def flow_conversion_bounds(index_values, index_names):
            """Return bounds of carrier_flow for bigM expression.

            :param index_values: list of index values
            :param index_names: list of index names
            :return: bounds: bounds of carrier_flow
            """
            # init the bounds
            index_arrs = self.zen_model.sets.tuple_to_arr(index_values, index_names)
            coords = [
                self.zen_model.sets.get_coord(data, name)
                for data, name in zip(index_arrs, index_names, strict=False)
            ]
            lower = xr.DataArray(0.0, coords=coords)
            upper = xr.DataArray(np.inf, coords=coords)

            # get the sets
            technology_set, carrier_set, node_set, timestep_set = [
                self.zen_model.sets[name] for name in index_names
            ]

            for tech in technology_set:
                for carrier in carrier_set[tech]:
                    time_step_year = [
                        self.time_steps.convert_time_step_operation2year(t)
                        for t in timestep_set
                    ]
                    if (
                        carrier
                        == self.zen_model.sets["set_reference_carriers"][tech][0]
                    ):
                        conversion_factor_lower = 1
                        conversion_factor_upper = 1
                    else:
                        conversion_factor_lower = (
                            self.zen_model.parameters.conversion_factor.loc[
                                tech, carrier, node_set
                            ]
                            .min()
                            .data
                        )
                        conversion_factor_upper = (
                            self.zen_model.parameters.conversion_factor.loc[
                                tech, carrier, node_set
                            ]
                            .max()
                            .data
                        )
                        if 0 in conversion_factor_upper:
                            _rounding_tsa = (
                                self.config.solver.rounding_decimal_points_tsa
                            )
                            raise ValueError(
                                f"Maximum conversion factor of {tech} for carrier "
                                f"{carrier} is 0.\n Potentially, the conversion factor "
                                f"is too small (1e-{_rounding_tsa}), so that it is "
                                f"rounded to 0 after the time series aggregation."
                            )

                    lower.loc[tech, carrier, ...] = (
                        self.zen_model.variables["capacity"]
                        .lower.loc[tech, "power", node_set, time_step_year]
                        .data
                        * conversion_factor_lower
                    )
                    upper.loc[tech, carrier, ...] = (
                        self.zen_model.variables["capacity"]
                        .upper.loc[tech, "power", node_set, time_step_year]
                        .data
                        * conversion_factor_upper
                    )

            # make sure lower is never below 0
            return lower, upper

        ## Flow variables
        # input flow of carrier into technology
        index_values, index_names = self.create_custom_set(
            [
                "set_conversion_technologies",
                "set_input_carriers",
                "set_nodes",
                "set_time_steps_operation",
            ],
        )
        self.zen_model.add_variable(
            name="flow_conversion_input",
            index_sets=(index_values, index_names),
            bounds=flow_conversion_bounds(index_values, index_names),
            doc="Carrier input of conversion technologies",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # output flow of carrier into technology
        index_values, index_names = self.create_custom_set(
            [
                "set_conversion_technologies",
                "set_output_carriers",
                "set_nodes",
                "set_time_steps_operation",
            ],
        )
        self.zen_model.add_variable(
            name="flow_conversion_output",
            index_sets=(index_values, index_names),
            bounds=flow_conversion_bounds(index_values, index_names),
            doc="Carrier output of conversion technologies",
            unit_category={"energy_quantity": 1, "time": -1},
        )

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for ConversionTechnology")

        for ConversionTechnologyConstraint in CONVERSION_TECHNOLOGY_CONSTRAINTS:
            self.service_container.build(ConversionTechnologyConstraint).build()

        # capex
        self.service_container.build(LinearCapexConstraint).build()
