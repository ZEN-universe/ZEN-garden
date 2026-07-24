"""Constructor for the ConversionTechnology elements."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import override

from zen_garden.elements.conversion_technology import ConversionTechnology
from zen_garden.elements.conversion_technology_rules import ConversionTechnologyRules
from zen_garden.elements.element_constructor import ElementConstructor

logger = logging.getLogger(__name__)


class ConversionTechnologyConstructor(ElementConstructor):
    element_class = ConversionTechnology

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
    def construct_params(self):
        logger.info("Constructing parameters for ConversionTechnology")
        # slope of linearly modeled capex
        capex_data, capex_units = self.get_capex_all_elements(
            index_names=[
                "set_conversion_technologies",
                "set_capex_linear",
                "set_nodes",
                "set_years",
            ],
        )
        self.zen_model.add_parameter(
            name="capex_specific_conversion",
            doc="Parameter specifying the slope of the capex if approximated linearly",
            data=capex_data,
            dict_of_units=capex_units,
        )
        # slope of linearly modeled conversion efficiencies
        self.add_parameter(
            name="conversion_factor",
            index_names=[
                "set_conversion_technologies",
                "set_dependent_carriers",
                "set_nodes",
                "set_time_steps_operation",
            ],
            doc="Parameter which specifies the conversion factor",
        )
        # minimum annual average capacity factor
        self.add_parameter(
            name="min_full_load_hours_fraction",
            index_names=[
                "set_conversion_technologies",
                "set_nodes",
                "set_years",
            ],
            doc="Minimum full load hours as a fraction of the total hours "
            "per planning period",
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
        ## pwa Variables - Capex
        # pwa capacity
        self.zen_model.add_variable(
            name="capacity_approximation",
            index_sets=self.create_custom_set(
                ["set_conversion_technologies", "set_nodes", "set_years"],
            ),
            bounds=(0, np.inf),
            doc="pwa variable for size of installed technology on edge i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # pwa capex technology
        self.zen_model.add_variable(
            name="capex_approximation",
            index_sets=self.create_custom_set(
                ["set_conversion_technologies", "set_nodes", "set_years"],
            ),
            bounds=(0, np.inf),
            doc="pwa variable for capex for installing technology on edge i and time t",
            unit_category={"money": 1},
        )

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for ConversionTechnology")

        # add pwa constraints
        rules = ConversionTechnologyRules(
            self.config, self.zen_model, self.energy_system, self.time_steps
        )
        # capacity factor constraint
        rules.constraint_capacity_factor_conversion()
        # opex and emissions constraint for conversion technologies
        rules.constraint_opex_emissions_technology_conversion()
        # conversion factor
        rules.constraint_carrier_conversion()
        # minimum average annual capacity factor
        rules.constraint_minimum_full_load_hours()

        # capex
        set_pwa_capex = self.create_custom_set(
            [
                "set_conversion_technologies",
                "set_capex_pwa",
                "set_nodes",
                "set_years",
            ],
        )
        set_linear_capex = self.create_custom_set(
            [
                "set_conversion_technologies",
                "set_capex_linear",
                "set_nodes",
                "set_years",
            ],
        )
        if len(set_pwa_capex[0]) > 0:
            # if set_pwa_capex contains technologies:
            pwa_breakpoints, pwa_values = self.calculate_capex_pwa_breakpoints_values(
                set_pwa_capex[0]
            )
            self.zen_model.constraints.add_pw_constraint(
                index_values=set_pwa_capex[0],
                yvar="capex_approximation",
                xvar="capacity_approximation",
                break_points=pwa_breakpoints,
                f_vals=pwa_values,
                cons_type="EQ",
                name="constraint_capex_pwa",
            )
        if set_linear_capex[0]:
            # if set_linear_capex contains technologies:
            rules.constraint_linear_capex()
        # Coupling constraints
        rules.constraint_capacity_capex_coupling()

    def calculate_capex_pwa_breakpoints_values(self, set_pwa):
        """Calculates breakpoints and function values for piecewise affine constraint.
        Args:
            set_pwa: Set of variable indices in capex approximation for
            which pwa is performed.
        Returns:
            pwa_breakpoints: Dict of pwa breakpoint values indexed by variable indices.
            pwa_values: Dict of pwa function values indexed by variable indices.
        """
        pwa_breakpoints = {}
        pwa_values = {}

        # iterate through pwa variable's indices
        for index in set_pwa:
            pwa_breakpoints[index] = []
            pwa_values[index] = []
            if len(index) > 1:
                tech = index[0]
            else:
                tech = index
            # retrieve pwa variables
            pwa_parameter = self.element_registry.get_attribute_of_specific_element(
                self.element_class, tech, "pwa_capex"
            )
            pwa_breakpoints[index] = pwa_parameter["capacity_addition"]
            pwa_values[index] = pwa_parameter["capex"]
        return pwa_breakpoints, pwa_values

    def get_capex_all_elements(self, index_names: list[str]):
        """Similar to Element.get_attribute_of_all_elements but only for capex.
        If select_pwa, extract pwa attributes, otherwise linear.

        :param index_names: list of index names
        :return: dict_of_attributes: returns dict of attribute values
        """
        class_elements = self.element_registry.all_elements_of_type(
            ConversionTechnology
        )
        dict_of_attributes = {}
        dict_of_units = {}
        is_pwa_attribute = "capex_is_pwa"
        attribute_name_linear = "capex_specific_conversion"

        for element in class_elements:
            # extract for pwa
            if not getattr(element, is_pwa_attribute):
                dict_of_attributes, _, dict_of_units = (
                    self.element_registry.append_attribute_of_element_to_dict(
                        element,
                        attribute_name_linear,
                        dict_of_attributes,
                        dict_of_units=dict_of_units,
                    )
                )

        if not dict_of_attributes:
            _, index_names = self.create_custom_set(index_names)
            return (dict_of_attributes, index_names), dict_of_units

        new_dict_of_attributes = pd.concat(
            dict_of_attributes, keys=list(dict_of_attributes.keys())
        )

        if not index_names:
            raise ValueError(
                "Initializing the parameter capex without the specifying the "
                "index names is not possible anymore!",
            )

        custom_set, index_names = self.create_custom_set(index_names)
        new_dict_of_attributes = self._check_for_subindex(
            new_dict_of_attributes, custom_set
        )
        return (new_dict_of_attributes, index_names), dict_of_units
