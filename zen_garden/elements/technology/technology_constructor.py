"""Constructor for the Technology elements."""

import logging

import numpy as np
import xarray as xr
from typing_extensions import override

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.technology import Technology
from zen_garden.elements.technology.constraints import (
    TECHNOLOGY_CONSTRAINTS,
    TechnologyOnOffConstraint,
)
from zen_garden.model.components.multi_index_helper import MultiIndexHelper
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
    def construct_vars(self):
        logger.info("Constructing variables for Technology")

        # TODO: This could be vectorized
        def capacity_bounds(tech, capacity_type, loc, time):
            """Return bounds of capacity for bigM expression.

            :param tech: tech index
            :param capacity_type: either power or energy
            :param loc: location of capacity
            :param time: investment time step
            :return: bounds: bounds of capacity
            """
            # bounds only needed for Big-M formulation,
            #   thus if any technology is modeled with on-off behavior
            if tech in techs_on_off_flag:
                params = self.zen_model.parameters.dict_parameters
                capacity_existing = params.capacity_existing
                capacity_addition_max = params.capacity_addition_max
                capacity_limit = params.capacity_limit
                capacities_existing = 0
                for id_technology_existing in self.zen_model.sets[
                    "set_technologies_existing"
                ][tech]:
                    if (
                        params.lifetime_existing[tech, loc, id_technology_existing]
                        > params.lifetime[tech]
                    ):
                        if (
                            time
                            > params.lifetime_existing[
                                tech, loc, id_technology_existing
                            ]
                            - params.lifetime[tech]
                        ):
                            capacities_existing += capacity_existing[
                                tech, capacity_type, loc, id_technology_existing
                            ]
                    elif (
                        time
                        <= params.lifetime_existing[tech, loc, id_technology_existing]
                        + 1
                    ):
                        capacities_existing += capacity_existing[
                            tech, capacity_type, loc, id_technology_existing
                        ]

                capacity_addition_max = (
                    len(self.zen_model.sets["set_years"])
                    * capacity_addition_max[tech, capacity_type]
                )
                max_capacity_limit = capacity_limit[tech, capacity_type, loc, time]
                bound_capacity = min(
                    capacity_addition_max + capacities_existing,
                    max_capacity_limit + capacities_existing,
                )
                return 0, bound_capacity
            else:
                return 0, np.inf

        techs_on_off, index_list = self.create_custom_set(
            [
                "set_technologies",
                "set_on_off",
                "set_location",
                "set_time_steps_operation",
            ],
        )
        index_list.pop(1)
        mask_on_off = self.zen_model.sets.indices_to_mask(
            techs_on_off, index_list, (0, 0)
        )[0]
        times = self.zen_model.sets["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [self.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
            dims=["set_time_steps_operation"],
        )
        mask_nonzero_cap_limit = (
            self.zen_model.parameters.capacity_limit.sel(
                {"set_capacity_types": "power", "set_years": time_step_year}
            )
            != 0
        )
        mask_on_off = mask_on_off & mask_nonzero_cap_limit.drop_vars(
            "set_capacity_types"
        )

        for variable in self.variables:
            if variable.name in [
                "carbon_emissions_technology_total",
                "cost_opex_yearly_total",
                "cost_capex_yearly_total",
            ]:
                # Exceptional bounds, masks or indices
                index_sets = self.zen_model.sets["set_years"]
                bounds = variable.get_bounds()
                mask = None

            elif variable.name in ["capacity"]:
                techs_on_off_flag = self.create_custom_set(
                    ["set_technologies", "set_on_off"]
                )[0]
                index_sets = self.create_custom_set(variable.indices)
                bounds = capacity_bounds
                mask = None
            elif variable.name in ["technology_installation"]:
                # Note: binary variables are written into the lp file by linopy even
                # if they are not relevant for the optimization, which makes all
                # problems MIPs. Therefore, we only add binary variables if really
                # necessary. Gurobi can handle this by noting that the binary
                # variables are not part of the model. However, only if there are no
                # binary variables at all is it possible to get the dual values of
                # the constraints.
                index_sets = self.create_custom_set(variable.indices)
                bounds = variable.get_bounds()
                mask = self._technology_installation_mask()
                if not mask.any():
                    continue
            elif variable.name in ["capacity_on_off_helper_var", "tech_on_var"]:
                index_sets = self.create_custom_set(variable.indices)
                bounds = variable.get_bounds()
                mask = mask_on_off
            else:
                # Standard behavior
                index_sets = self.create_custom_set(variable.indices)
                bounds = variable.get_bounds()
                mask = None

            self.zen_model.add_variable(
                name=variable.name,
                index_sets=index_sets,
                binary=variable.binary,
                bounds=bounds,
                mask=mask,
                doc=variable.doc,
                unit_category=variable.unit_category,
            )

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for Technology")

        for TechnologyConstraint in TECHNOLOGY_CONSTRAINTS:
            self.service_container.build(TechnologyConstraint).build()

        # min load constraints
        n_cons = len(self.zen_model.lp_model.constraints.items())
        self.service_container.build(TechnologyOnOffConstraint).build()
        # if nothing was added we can remove the tech vars again
        if len(self.zen_model.lp_model.constraints.items()) == n_cons:
            self.zen_model.lp_model.variables.remove("tech_on_var")
            self.zen_model.lp_model.variables.remove("capacity_on_off_helper_var")

    def _technology_installation_mask(self) -> xr.DataArray:
        """Check if the binary variable is necessary."""
        mask = xr.DataArray(
            False,
            coords=[
                self.zen_model.lp_model.variables.coords["set_years"],
                self.zen_model.lp_model.variables.coords["set_technologies"],
                self.zen_model.lp_model.variables.coords["set_capacity_types"],
                self.zen_model.lp_model.variables.coords["set_location"],
            ],
        )

        # used in transport technology
        techs = list(self.zen_model.sets["set_transport_technologies"])
        if len(techs) > 0:
            edges = list(self.zen_model.sets["set_edges"])
            sub_mask = (
                self.zen_model.parameters.distance.loc[techs, edges]
                * self.zen_model.parameters.capex_per_distance_transport.loc[
                    techs, edges
                ]
                != 0
            )
            sub_mask = sub_mask.rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
            mask.loc[:, techs, :, edges] |= sub_mask

        # used in constraint_technology_min_capacity_addition
        mask = mask | (
            self.zen_model.parameters.capacity_addition_min.notnull()
            & (self.zen_model.parameters.capacity_addition_min != 0)
        )

        # used in constraint_technology_max_capacity_addition
        index_values, index_names = self.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ],
        )
        index = MultiIndexHelper(index_values, index_names)
        sub_mask = (
            self.zen_model.parameters.capacity_addition_max.notnull()
            & (self.zen_model.parameters.capacity_addition_max != np.inf)
            & (self.zen_model.parameters.capacity_addition_max != 0)
        )
        for tech, capacity_type in index.get_unique([0, 1]):
            locs = index.get_values(locs=[tech, capacity_type], levels=2, unique=True)
            mask.loc[:, tech, capacity_type, locs] |= sub_mask.loc[tech, capacity_type]

        return mask

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
