"""Constructor for the Technology elements."""

import logging

import numpy as np
import xarray as xr
from typing_extensions import override

from zen_garden.constraints.technology import (
    TECHNOLOGY_CONSTRAINTS,
    CostCapexYearlyConstraint,
    TechnologyDiffusionLimitConstraint,
)
from zen_garden.constraints.technology.technology_on_off_constraint import (
    TechnologyOnOffConstraint,
)
from zen_garden.elements.element_constructor import ElementConstructor
from zen_garden.elements.technology import Technology
from zen_garden.model.components.index_set import IndexSet
from zen_garden.model.components.zen_index import ZenIndex

logger = logging.getLogger(__name__)


class TechnologyConstructor(ElementConstructor):
    element_class = Technology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.technology.Technology`.

        :return: True if there are elements, False otherwise
        """
        return True

    ### --- classmethods to construct sets, parameters, variables, and constraints,
    # that correspond to Technology --- ###
    @override
    def construct_sets(self):
        logger.info("Constructing sets for Technology")

        # conversion technologies
        self.zen_model.add_set(
            name="set_conversion_technologies",
            data=self.energy_system.set_conversion_technologies,
            doc="Set of conversion technologies",
        )
        # retrofitting technologies
        self.zen_model.add_set(
            name="set_retrofitting_technologies",
            data=self.energy_system.set_retrofitting_technologies,
            doc="Set of retrofitting technologies",
        )
        # transport technologies
        self.zen_model.add_set(
            name="set_transport_technologies",
            data=self.energy_system.set_transport_technologies,
            doc="Set of transport technologies",
        )
        # storage technologies
        self.zen_model.add_set(
            name="set_storage_technologies",
            data=self.energy_system.set_storage_technologies,
            doc="Set of storage technologies",
        )
        # existing installed technologies
        self.zen_model.add_set(
            name="set_technologies_existing",
            data=self.element_registry.get_attribute_of_all_elements(
                self.element_class, "set_technologies_existing"
            ),
            doc="Set of existing technologies",
            index_set="set_technologies",
        )
        # reference carriers
        self.zen_model.add_set(
            name="set_reference_carriers",
            data=self.element_registry.get_attribute_of_all_elements(
                self.element_class, "reference_carrier"
            ),
            doc="set of all reference carriers correspondent to a technology. "
            "Indexed by set_technologies",
            index_set="set_technologies",
        )

    @override
    def construct_params(self):
        logger.info("Constructing parameters for Technology")

        # existing capacity
        self.add_parameter(
            name="capacity_existing",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_technologies_existing",
            ],
            capacity_types=True,
            doc="Parameter which specifies the existing technology size",
        )
        # existing capacity
        self.add_parameter(
            name="capacity_investment_existing",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years_entire_horizon",
            ],
            capacity_types=True,
            doc="Parameter specifying the size of the previously invested capacities",
        )
        # minimum capacity addition
        self.add_parameter(
            name="capacity_addition_min",
            index_names=["set_technologies", "set_capacity_types"],
            capacity_types=True,
            doc="Parameter which specifies the minimum capacity addition "
            "that can be installed",
        )
        # maximum capacity addition
        self.add_parameter(
            name="capacity_addition_max",
            index_names=["set_technologies", "set_capacity_types"],
            capacity_types=True,
            doc="Parameter which specifies the maximum capacity addition "
            "that can be installed",
        )
        # unbounded capacity addition
        self.add_parameter(
            name="capacity_addition_unbounded",
            index_names=["set_technologies"],
            doc="Parameter which specifies the unbounded capacity addition that can be "
            "added each year (only for delayed technology deployment)",
        )
        # lifetime existing technologies
        self.add_parameter(
            name="lifetime_existing",
            index_names=[
                "set_technologies",
                "set_location",
                "set_technologies_existing",
            ],
            doc="Parameter specifying the remaining lifetime of an existing technology",
        )
        # lifetime existing technologies
        self.add_parameter(
            name="capex_capacity_existing",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_technologies_existing",
            ],
            capacity_types=True,
            doc="Parameter which specifies the total capex of an existing technology "
            "which still has to be paid",
        )
        # variable specific opex
        self.add_parameter(
            name="opex_specific_variable",
            index_names=[
                "set_technologies",
                "set_location",
                "set_time_steps_operation",
            ],
            doc="Parameter which specifies the variable specific opex",
        )
        # fixed specific opex
        self.add_parameter(
            name="opex_specific_fixed",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ],
            capacity_types=True,
            doc="Parameter which specifies the fixed annual specific opex",
        )
        # lifetime newly built technologies
        self.add_parameter(
            name="lifetime",
            index_names=["set_technologies"],
            doc="Parameter which specifies the lifetime of a newly built technology",
        )
        # amortization time newly built technologies
        self.add_parameter(
            name="depreciation_time",
            index_names=["set_technologies"],
            doc="Parameter which specifies the depreciation time of a "
            "newly built technology",
        )
        # construction_time newly built technologies
        self.add_parameter(
            name="construction_time",
            index_names=["set_technologies"],
            doc="Parameter which specifies the construction time of a "
            "newly built technology",
        )
        # maximum diffusion rate, i.e., increase in capacity
        self.add_parameter(
            name="max_diffusion_rate",
            index_names=["set_technologies", "set_years"],
            doc="Parameter which specifies the maximum diffusion rate which is the "
            "maximum increase in capacity between investment steps",
        )
        # capacity_limit of technologies
        self.add_parameter(
            name="capacity_limit",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ],
            capacity_types=True,
            doc="Parameter which specifies the capacity limit of technologies",
        )
        # NEW: lower capacity limit of technologies
        self.add_parameter(
            name="capacity_lower_limit",
            index_names=[
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ],
            capacity_types=True,
            doc="Parameter which specifies the lower capacity limit of technologies",
        )
        # minimum load relative to capacity
        self.add_parameter(
            name="min_load",
            index_names=[
                "set_technologies",
                "set_location",
                "set_time_steps_operation",
            ],
            doc="Parameter which specifies the minimum load of technology "
            "relative to installed capacity",
        )
        # maximum load relative to capacity
        self.add_parameter(
            name="max_load",
            index_names=[
                "set_technologies",
                "set_location",
                "set_time_steps_operation",
            ],
            doc="Parameter which specifies the maximum load of technology relative to "
            "installed capacity",
        )
        # carbon intensity
        self.add_parameter(
            name="carbon_intensity_technology",
            index_names=["set_technologies", "set_location"],
            doc="Parameter which specifies the carbon intensity of each technology",
        )
        # calculate additional existing parameters
        self.zen_model.add_parameter(
            name="existing_capacities",
            data=self.get_existing_quantity("capacity"),
            doc="Parameter which specifies the total available capacity of existing "
            "technologies at the beginning of the optimization",
        )
        self.zen_model.add_parameter(
            name="existing_capex",
            data=self.get_existing_quantity("cost_capex_overnight"),
            doc="Parameter which specifies the total capex of existing technologies at "
            "the beginning of the optimization",
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
            if tech in techs_on_off:
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

        # bounds only needed for Big-M formulation,
        #   thus if any technology is modeled with on-off behavior
        techs_on_off = self.create_custom_set(["set_technologies", "set_on_off"])[0]
        # construct pe.Vars of the class <Technology>
        # capacity technology
        self.zen_model.add_variable(
            name="capacity",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_years",
                ],
            ),
            bounds=capacity_bounds,
            doc="size of installed technology at location l and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # capacity technology before current year
        self.zen_model.add_variable(
            name="capacity_previous",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_years",
                ],
            ),
            bounds=(0, np.inf),
            doc="size of installed technology at location l and BEFORE time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # built_capacity technology
        self.zen_model.add_variable(
            name="capacity_addition",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_years",
                ],
            ),
            bounds=(0, np.inf),
            doc="size of built technology (invested capacity after construction) "
            "at location l and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # invested_capacity technology
        self.zen_model.add_variable(
            name="capacity_investment",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_years",
                ],
            ),
            bounds=(0, np.inf),
            doc="size of invested technology at location l and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # capex of building capacity overnight
        self.zen_model.add_variable(
            name="cost_capex_overnight",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_years",
                ],
            ),
            bounds=(0, np.inf),
            doc="capex for building technology at location l and time t",
            unit_category={"money": 1},
        )
        # annual capex of having capacity
        self.zen_model.add_variable(
            name="cost_capex_yearly",
            index_sets=self.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types",
                    "set_location",
                    "set_years",
                ],
            ),
            bounds=(0, np.inf),
            doc="annual capex for having technology at location l",
            unit_category={"money": 1},
        )
        # total capex
        self.zen_model.add_variable(
            name="cost_capex_yearly_total",
            index_sets=self.zen_model.sets["set_years"],
            bounds=(0, np.inf),
            doc="total capex for installing all technologies in all locations "
            "at all times",
            unit_category={"money": 1},
        )
        # opex
        self.zen_model.add_variable(
            name="cost_opex_variable",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
            ),
            bounds=(0, np.inf),
            doc="opex for operating technology at location l and time t",
            unit_category={"money": 1, "time": -1},
        )
        # total opex
        self.zen_model.add_variable(
            name="cost_opex_yearly_total",
            index_sets=self.zen_model.sets["set_years"],
            bounds=(0, np.inf),
            doc="total opex all technologies and locations in year y",
            unit_category={"money": 1},
        )
        # yearly opex
        self.zen_model.add_variable(
            name="cost_opex_yearly",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_years"],
            ),
            bounds=(0, np.inf),
            doc="yearly opex for operating technology at location l and year y",
            unit_category={"money": 1},
        )
        # carbon emissions
        self.zen_model.add_variable(
            name="carbon_emissions_technology",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
            ),
            doc="carbon emissions for operating technology at location l and time t",
            unit_category={"emissions": 1, "time": -1},
        )
        # total carbon emissions technology
        self.zen_model.add_variable(
            name="carbon_emissions_technology_total",
            index_sets=self.zen_model.sets["set_years"],
            doc="total carbon emissions for operating technology",
            unit_category={"emissions": 1},
        )

        # install technology
        # Note: binary variables are written into the lp file by linopy even if they
        # are not relevant for the optimization, which makes all problems MIPs.
        # Therefore, we only add binary variables, if really necessary. Gurobi can
        # handle this by noting that the binary variables are not part of the model,
        # however, only if there are no binary variables at all, it is possible to get
        # the dual values of the constraints.
        mask = self._technology_installation_mask()
        if mask.any():
            self.zen_model.add_variable(
                name="technology_installation",
                index_sets=self.create_custom_set(
                    [
                        "set_technologies",
                        "set_capacity_types",
                        "set_location",
                        "set_years",
                    ],
                ),
                binary=True,
                doc="installment of a technology at location l and time t",
                mask=mask,
                unit_category=None,
            )

        # on-off variables
        # We remove the binary variables if there are any no constraints that use them
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
        self.zen_model.add_variable(
            name="tech_on_var",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
            ),
            mask=mask_on_off,
            doc="Binary variable which equals 1 when technology is switched on at "
            "location l and time t",
            binary=True,
            unit_category=None,
        )
        self.zen_model.add_variable(
            name="capacity_on_off_helper_var",
            index_sets=self.create_custom_set(
                ["set_technologies", "set_location", "set_time_steps_operation"],
            ),
            bounds=(0, np.inf),
            mask=mask_on_off,
            doc="Helper variable substituting the product of capacity and tech_on_var",
            unit_category={"energy_quantity": 1, "time": -1},
        )

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for Technology")

        for TechnologyConstraint in TECHNOLOGY_CONSTRAINTS:
            TechnologyConstraint(
                self.config, self.zen_model, self.energy_system, self.time_steps
            ).build()

        TechnologyDiffusionLimitConstraint(
            self.config,
            self.zen_model,
            self.energy_system,
            self.time_steps,
            self.element_registry,
        ).build()

        # annual capex of having capacity
        index_values, index_names = self.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ],
        )
        CostCapexYearlyConstraint(
            self.config, self.zen_model, self.energy_system, self.time_steps
        ).build(ZenIndex(index_values, index_names))

        # min load constraints
        n_cons = len(self.zen_model.lp_model.constraints.items())
        techs_on_off = self.create_custom_set(["set_technologies", "set_on_off"])[0]
        # rules.constraint_technology_on_off(techs_on_off)
        TechnologyOnOffConstraint(
            self.config, self.zen_model, self.energy_system, self.time_steps
        ).build(techs_on_off)

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
        index = ZenIndex(index_values, index_names)
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
        index_arrs = IndexSet.tuple_to_arr(index_values, index_names)
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
        current_year_horizon = self.energy_system.set_years[0]
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
