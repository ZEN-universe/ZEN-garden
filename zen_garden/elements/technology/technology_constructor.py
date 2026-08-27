"""Constructor for the Technology elements."""

import logging

import numpy as np
import xarray as xr
from typing_extensions import override

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.technology import Technology
from zen_garden.elements.technology.constraints import TechnologyOnOffConstraint
from zen_garden.model.components.set_registry import SetRegistry

logger = logging.getLogger(__name__)


class TechnologyConstructor(ModelConstructor):
    element_class = Technology

    @override
    def construct_params(self):
        """Construct technology parameters and calculated existing quantities."""
        super().construct_params()
        self.zen_model.add_parameter(
            name="existing_capacities",
            data=self.get_existing_quantity("capacity"),
            doc="Total available existing capacity at the optimization start",
        )
        self.zen_model.add_parameter(
            name="existing_capex",
            data=self.get_existing_quantity("cost_capex_overnight"),
            doc="Total capex of existing technologies at the optimization start",
        )

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for Technology")

        for TechnologyConstraint in self.constraints:
            self.service_container.build(TechnologyConstraint).build()

        # min load constraints (built last, with special-case cleanup)
        n_cons = len(self.zen_model.lp_model.constraints.items())
        self.service_container.build(TechnologyOnOffConstraint).build()
        # if nothing was added we can remove the tech vars again
        if len(self.zen_model.lp_model.constraints.items()) == n_cons:
            for variable_name in ("tech_on_var", "capacity_on_off_helper_var"):
                if variable_name in self.zen_model.lp_model.variables:
                    self.zen_model.lp_model.variables.remove(variable_name)
    def get_existing_quantity(self, type_existing_quantity: str):
        """Get existing capacities of all technologies.

        :param type_existing_quantity: capacity or cost_capex_overnight
        :return: The existing capacities
        """
        index_values, index_names = self.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ]
        )
        # get all the capacities
        index_arrs = SetRegistry.tuple_to_arr(index_values, index_names)
        coords = [
            self.zen_model.sets.get_coord(data, name)
            for data, name in zip(index_arrs, index_names, strict=False)
        ]
        existing_quantities = xr.DataArray(np.nan, coords=coords, dims=index_names)
        values = np.zeros(len(index_values))
        for i, (tech, capacity_type, loc, time) in enumerate(index_values):
            values[i] = self._get_available_existing_quantity(
                tech,
                capacity_type,
                loc,
                time,
                type_existing_quantity,
            )
        existing_quantities.loc[index_arrs] = values
        return existing_quantities

    def _get_available_existing_quantity(
        self, tech, capacity_type, loc, year, type_existing_quantity
    ):
        """Gets the existing quantity of 'tech' at investment time step 'time'.

        returns existing quantity of 'tech', that is still available at invest
        time step 'time'. Either capacity or capex.

        :param tech: name of technology
        :param capacity_type: type of capacity
        :param loc: location (node or edge) of existing capacity
        :param year: current yearly time step
        :param type_existing_quantity: capex or capacity
        :return: existing_quantity: existing capacity or capex of existing capacity
        """
        params = self.zen_model.parameters.dict_parameters
        existing_quantity = 0
        if type_existing_quantity == "capacity":
            existing_variable = params.capacity_existing
        elif type_existing_quantity == "cost_capex_overnight":
            existing_variable = params.capex_capacity_existing
        else:
            raise KeyError(f"Wrong type of existing quantity {type_existing_quantity}")

        for id_capacity_existing in self.zen_model.sets["set_technologies_existing"][
            tech
        ]:
            is_existing = self.get_if_capacity_still_existing(
                tech,
                year,
                loc=loc,
                id_capacity_existing=id_capacity_existing,
            )
            # if still available at first base time step, add to list
            if is_existing:
                existing_quantity += existing_variable[
                    tech, capacity_type, loc, id_capacity_existing
                ]
        return existing_quantity

    def get_if_capacity_still_existing(self, tech, year, loc, id_capacity_existing):
        """Returns boolean if capacity still exists at yearly time step 'year'.

        :param tech: name of technology
        :param year: yearly time step
        :param loc: location
        :param id_capacity_existing: id of existing capacity
        :return: boolean if still existing
        """
        # get params and system
        params = self.zen_model.parameters.dict_parameters
        # get lifetime of existing capacity
        lifetime_existing = params.lifetime_existing[tech, loc, id_capacity_existing]
        lifetime = params.lifetime[tech]
        delta_lifetime = lifetime_existing - lifetime
        # reference year of current optimization horizon
        current_year_horizon = self.model_schema.set_years[0]
        if delta_lifetime >= 0:
            cutoff_year = (
                year - current_year_horizon
            ) * self.config.system.interval_between_years
            return cutoff_year >= delta_lifetime
        else:
            cutoff_year = (
                year - current_year_horizon + 1
            ) * self.config.system.interval_between_years
            return cutoff_year <= lifetime_existing
