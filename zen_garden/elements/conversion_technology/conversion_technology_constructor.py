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
    sets = ConversionTechnology.own_sets
    variables = ConversionTechnology.variables

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.conversion_technology.ConversionTechnology`.

        :return: True if there are elements, False otherwise
        """
        return True

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


        for variable in self.variables:
            if variable.name in ["flow_conversion_input", "flow_conversion_output"]:
                # Exceptional bounds, masks or indices
                index_values, index_names = self.create_custom_set(variable.indices)
                index_sets = (index_values, index_names)
                bounds = flow_conversion_bounds(index_values, index_names)
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
    def construct_constraints(self):
        logger.info("Constructing constraints for ConversionTechnology")

        for ConversionTechnologyConstraint in CONVERSION_TECHNOLOGY_CONSTRAINTS:
            self.service_container.build(ConversionTechnologyConstraint).build()

        # capex
        self.service_container.build(LinearCapexConstraint).build()
