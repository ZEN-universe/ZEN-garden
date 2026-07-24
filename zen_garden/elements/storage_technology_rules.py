"""Rules for the StorageTechnology class."""

import logging

import numpy as np
import xarray as xr

from zen_garden.elements.generic_rule import GenericRule
from zen_garden.model.components.index_set import IndexSet
from zen_garden.utils import linexpr_from_tuple_np

logger = logging.getLogger(__name__)


class StorageTechnologyRules(GenericRule):
    """Rules for the StorageTechnology class."""

    def constraint_charge_discharge_binary(self):
        """Avoid simultaneous charge and discharge of storage technologies.

        Ensure that the storage technology cannot charge and discharge simultaneously
        within the same operational time step. This is only active if the
        storage_charge_discharge_binary flag in the system.json file is set to True.

        .. math::
            \\underline{H}_{k,n,t,y} \\le B^\\mathrm{charge}_{k,n,t,y}
            s^\\mathrm{max, power}_{k,n,y}

        .. math::
            \\overline{H}_{k,n,t,y} \\le (1-B^\\mathrm{charge}_{k,n,t,y})
            s^\\mathrm{max, power}_{k,n,y}

        :math:`\\underline{H}_{k,n,t,y}`: carrier flow into storage technology :math:`k`
        on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`\\overline{H}_{k,n,t,y}`: carrier flow out of storage
        technology :math:`k`on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`s^\\mathrm{max, power}_{k,n,y}`: power capacity limit of storage
        technology :math:`k` at location :math:`n` in year :math:`y` \n
        :math:`B^\\mathrm{charge}_{k,n,t,y}`: binary variable indicating whether the
        storage technology :math:`k` is in charging mode (1) or discharging mode (0) at
        location :math:`n` at time step :math:`t` in year :math:`y` \n
        """
        techs = self.zen_model.sets["set_storage_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        if len(techs) == 0:
            return
        # capacity limit as upper bound
        times = self.get_storage2year_time_step_array()
        capacity_limit = self.zen_model.parameters.capacity_limit
        capacity_limit = self.map_and_expand(capacity_limit, times)
        capacity_limit = capacity_limit.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        capacity_limit = capacity_limit.sel(
            {
                "set_nodes": nodes,
                "set_storage_technologies": techs,
                "set_capacity_types": "power",
            }
        )
        capacity_limit = capacity_limit.rename(
            {"set_time_steps_storage": "set_time_steps_operation"}
        )

        lhs = (
            self.zen_model.lp_model.variables["flow_storage_charge"]
            - self.zen_model.lp_model.variables["charge_storage_binary"]
            * capacity_limit
        )
        rhs = 0
        constraint_charge = lhs <= rhs

        lhs = (
            self.zen_model.lp_model.variables["flow_storage_discharge"]
            + self.zen_model.lp_model.variables["charge_storage_binary"]
            * capacity_limit
        )
        rhs = capacity_limit
        constraint_discharge = lhs <= rhs

        self.zen_model.add_constraint(
            "constraint_charge_storage_binary", constraint_charge
        )
        self.zen_model.add_constraint(
            "constraint_discharge_storage_binary", constraint_discharge
        )

    def constraint_capacity_factor_storage(self):
        """Limits load of storage technologies by capacity and maximum load factor.

        .. math::
            \\underline{H}_{k,n,t,y}+\\overline{H}_{k,n,t,y}\\leq
            m^{\\mathrm{max}}_{k,n,t,y}S_{k,n,y}

        :math:`\\underline{H}_{k,n,t,y}`: carrier flow into storage technology :math:`k`
        on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`\\overline{H}_{k,n,t,y}`: carrier flow out of storage
        technology :math:`k`on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`m^{\\mathrm{max}}_{k,n,t,y}`: maximum load factor for storage
        technology :math:`k` on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`S_{k,n,y}`: storage capacity of storage technology :math:`k` on
        node :math:`n` in year :math:`y`


        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        times = self.zen_model.lp_model.variables.coords["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [self.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
        )
        term_capacity = (
            self.zen_model.parameters.max_load.loc[techs, nodes, :]
            * self.zen_model.lp_model.variables["capacity"].loc[
                techs, "power", nodes, time_step_year
            ]
        ).rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )

        # TODO integrate level storage here as well
        lhs = term_capacity - self.get_flow_expression_storage(rename=False)
        rhs = 0
        constraints = lhs >= rhs
        ### return
        self.zen_model.add_constraint("constraint_capacity_factor_storage", constraints)

    def constraint_opex_emissions_technology_storage(self):
        """Calculate opex of each technology.

        .. math::
            O_{h,p,t}^\\mathrm{t} = \\beta_{h,p,t} (\\underline{H}_{k,n,t} +
            \\overline{H}_{k,n,t}) \n
            \\theta_{h,p,t}^{\\mathrm{tech}} = \\epsilon_h (\\underline{H}_{k,n,t} +
            \\overline{H}_{k,n,t})

        :math:`O_{h,p,t}^\\mathrm{t}`: variable operational expenditures for storage
        technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\beta_{h,p,t}`: specific variable operational expenditures for storage
        technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\underline{H}_{k,n,t}`: carrier flow into storage technology :math:`k`
        on node :math:`n` and time :math:`t` \n
        :math:`\\overline{H}_{k,n,t}`: carrier flow out of storage technology :math:`k`
        on node :math:`n` and time :math:`t` \n
        :math:`\\theta_{h,p,t}^{\\mathrm{tech}}`: carbon emissions for storage
        technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\epsilon_h`: carbon intensity for operating storage technology :math:`h`
        on node :math:`n`

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        lhs_opex = self.zen_model.lp_model.variables["cost_opex_variable"].sel(
            {"set_technologies": techs, "set_location": nodes}
        ) - (
            self.zen_model.parameters.opex_specific_variable
            * self.get_flow_expression_storage()
        )
        lhs_emissions = self.zen_model.lp_model.variables[
            "carbon_emissions_technology"
        ].sel({"set_technologies": techs, "set_location": nodes}) - (
            self.zen_model.parameters.carbon_intensity_technology
            * self.get_flow_expression_storage()
        )
        lhs_opex = lhs_opex.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        lhs_emissions = lhs_emissions.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs

        self.zen_model.add_constraint(
            "constraint_opex_technology_storage", constraints_opex
        )
        self.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_storage", constraints_emissions
        )

    def constraint_storage_level_max(self):
        """Limit maximum storage level to capacity.

        .. math::
            L_{k,n,t^\\mathrm{k}} \\le S^\\mathrm{e}_{k,n,y}

        :math:`L_{k,n,t^\\mathrm{k}}`: storage level of storage technology :math:`k`
        on node :math:`n` and time :math:`t` \n
        :math:`S^\\mathrm{e}_{k,n,y}`: energy capacity of storage technology :math:`k`
        on node :math:`n` in year :math:`y`

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        if len(techs) == 0:
            return
        # mask for energy capacity and storage time steps
        times = self.get_storage2year_time_step_array()
        capacity = self.map_and_expand(
            self.zen_model.lp_model.variables["capacity"], times
        )
        capacity = capacity.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        capacity = capacity.sel({"set_nodes": nodes, "set_storage_technologies": techs})
        storage_level = self.zen_model.lp_model.variables["storage_level"]
        mask_capacity_type = (
            self.zen_model.lp_model.variables["capacity"].coords["set_capacity_types"]
            == "energy"
        )
        lhs = (storage_level - capacity).where(mask_capacity_type, 0.0)
        rhs = 0
        constraints = lhs <= rhs

        self.zen_model.add_constraint("constraint_storage_level_max", constraints)

    def constraint_capacity_energy_to_power_ratio(self):
        """Limit capacity power to energy ratio.

        .. math::
            \\rho_k^{min} S^{e}_{k,n,y} \\le S_{k,n,y}

        .. math::
            S_{k,n,y} \\le \\rho_k^{max} S^{e}_{k,n,y}

        :math:`S^{\\mathrm{power}}_{k,n,y}`: installed capacity in terms of power of
        storage :math:`k` at node :math:`n` in year :math:`y` \n
        :math:`S^{e}_{k,n,y}`: installed capacity in terms of energy of
        storage :math:`k` at node :math:`n` in year :math:`y` \n
        :math:`\\rho_k^{min}`: minimum power-to-energy ratio of storage :math:`k` \n
        :math:`\\rho_k^{max}`: maximum power-to-energy ratio of storage :math:`k`

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return None
        e2p_min = self.zen_model.parameters.energy_to_power_ratio_min
        e2p_max = self.zen_model.parameters.energy_to_power_ratio_max
        mask_min = e2p_min != np.inf
        mask_max = e2p_max != np.inf

        capacity_addition = self.zen_model.lp_model.variables[
            "capacity_addition"
        ].rename({"set_technologies": "set_storage_technologies"})
        capacity_addition_power = capacity_addition.sel(
            {"set_storage_technologies": techs, "set_capacity_types": "power"}
        )
        capacity_addition_energy = capacity_addition.sel(
            {"set_storage_technologies": techs, "set_capacity_types": "energy"}
        )
        lhs = (capacity_addition_energy - capacity_addition_power * e2p_min).where(
            mask_min
        )
        rhs = 0
        constraints_min = lhs >= rhs
        lhs = (capacity_addition_energy - capacity_addition_power * e2p_max).where(
            mask_max
        )
        constraints_max = lhs <= rhs

        self.zen_model.add_constraint(
            "constraint_capacity_energy_to_power_ratio_min", constraints_min
        )
        self.zen_model.add_constraint(
            "constraint_capacity_energy_to_power_ratio_max", constraints_max
        )

    def constraint_couple_storage_level(self):
        """Couple subsequent storage levels (time coupling constraints).

        .. math::
            L_{k,n,t^k,y} = L_{k,n,t^k-1,y} (1-\\phi_k)^{\\tau_{t^k}^k} +
            (\\underline{\\eta}_k \\underline{H}_{k,n,\\sigma(t^k),y} -
            \\frac{\\overline{H}_{k,n,\\sigma(t^k),y}}{\\overline{\\eta}_k})
            \\sum^{\\tau_{t^k}^k-1}_{\\tilde{t}^k=0} (1-\\phi_k)^{\\tilde{t}^k}

        :math:`L_{k,n,t^k,y}`: storage level of storage technology :math:`k` on
        node :math:`n` and time :math:`t^k` in year :math:`y` \n
        :math:`\\phi_k`: self discharge rate of storage technology :math:`k` \n
        :math:`\\tau_{t^k}^k`: duration of storage level time step of storage
        technology :math:`k` \n
        :math:`\\underline{\\eta}_k`: efficiency during charging of storage
        technology :math:`k` \n
        :math:`\\overline{\\eta}_k`: efficiency during discharging of storage
        technology :math:`k` \n
        :math:`\\underline{H}_{k,n,\\sigma(t^k),y}`: charge flow into storage
        technology :math:`k` on node :math:`n` and time :math:`\\sigma(t^k)` in
        year :math:`y` \n
        :math:`\\overline{H}_{k,n,\\sigma(t^k),y}`: discharge flow out of storage
        technology :math:`k` on node :math:`n` and time :math:`\\sigma(t^k)` in
        year :math:`y`

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        self_discharge = self.zen_model.parameters.self_discharge
        flow_storage_inflow = self.zen_model.parameters.flow_storage_inflow
        flow_storage_spillage = self.zen_model.lp_model.variables.flow_storage_spillage
        time_steps_storage_duration = (
            self.zen_model.parameters.time_steps_storage_duration
        )
        # reformulate self discharge multiplier as partial geometric series
        multiplier_w_discharge = (
            1 - (1 - self_discharge) ** time_steps_storage_duration
        ) / (1 - (1 - self_discharge))
        multiplier_wo_discharge = time_steps_storage_duration
        multiplier = multiplier_w_discharge.where(
            self_discharge != 0, 0.0
        ) + multiplier_wo_discharge.where(self_discharge == 0, 0.0)
        # time coupling to previous time step
        times_coupling, mask_coupling = self.get_previous_storage_time_step_array()
        self_discharge_previous = (1 - self_discharge) ** time_steps_storage_duration
        self_discharge_previous["set_time_steps_storage"] = times_coupling
        term_delta_storage_level = self.zen_model.lp_model.variables[
            "storage_level"
        ] - self_discharge_previous * self.zen_model.lp_model.variables[
            "storage_level"
        ].sel(
            {"set_time_steps_storage": times_coupling}
        )
        # charge and discharge flow
        times_year_time_step = self.get_year_time_step_array()
        efficiency_charge = (
            self.zen_model.parameters.efficiency_charge.broadcast_like(
                times_year_time_step
            )
            .where(times_year_time_step, 0.0)
            .sum("set_years")
        )
        efficiency_discharge = (
            self.zen_model.parameters.efficiency_discharge.broadcast_like(
                times_year_time_step
            )
            .where(times_year_time_step, 0.0)
            .sum("set_years")
        )
        term_flow_charge_discharge = (
            self.zen_model.lp_model.variables["flow_storage_charge"] * efficiency_charge
            - self.zen_model.lp_model.variables["flow_storage_discharge"].to_linexpr()
            / efficiency_discharge
            + flow_storage_inflow
            - flow_storage_spillage
        )
        times_power2energy = self.get_power2energy_time_step_array()
        term_flow_charge_discharge = self.map_and_expand(
            term_flow_charge_discharge, times_power2energy
        )
        term_flow_charge_discharge = term_flow_charge_discharge * multiplier
        # sum up all terms
        lhs = (term_delta_storage_level - term_flow_charge_discharge).where(
            mask_coupling, 0.0
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_couple_storage_level", constraints)

    def constraint_flow_storage_spillage(self):
        """Ensure that flow_energy_spillage is not greater than the flow_storage_inflow.

        .. math::

        Todo:

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return

        flow_storage_inflow = self.zen_model.parameters.flow_storage_inflow
        flow_storage_spillage = self.zen_model.lp_model.variables.flow_storage_spillage

        lhs = flow_storage_spillage - flow_storage_inflow
        rhs = 0
        constraints = lhs <= rhs

        self.zen_model.add_constraint("constraint_flow_storage_spillage", constraints)

    def constraint_storage_technology_capex(self, index_values, index_names):
        """Definition of the capital expenditures for the storage technology.

        .. math::
            CAPEX_{y,n,i} = \\Delta S_{h,p,y} \\alpha_{k,n,y}

        :math:`\\Delta S_{h,p,y}`: capacity addition of storage technology :math:`h`
        on node :math:`n` in year :math:`y` \n
        :math:`\\alpha_{k,n,y}`: specific capex of storage technology :math:`k` on
        node :math:`n` in year :math:`y`


        """
        # check if we need to continue
        if len(index_values) == 0:
            return []

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # get all the arrays and coords
        techs, capacity_types, nodes, times = IndexSet.tuple_to_arr(
            index_values, index_names, unique=True
        )
        coords = [
            self.zen_model.lp_model.variables.coords["set_storage_technologies"],
            self.zen_model.lp_model.variables.coords["set_capacity_types"],
            self.zen_model.lp_model.variables.coords["set_nodes"],
            self.zen_model.lp_model.variables.coords["set_years"],
        ]

        ### formulate constraint
        lhs = linexpr_from_tuple_np(
            [
                (
                    1.0,
                    self.zen_model.lp_model.variables["cost_capex_overnight"].loc[
                        techs, capacity_types, nodes, times
                    ],
                ),
                (
                    -self.zen_model.parameters.capex_specific_storage.loc[
                        techs, capacity_types, nodes, times
                    ],
                    self.zen_model.lp_model.variables["capacity_addition"].loc[
                        techs, capacity_types, nodes, times
                    ],
                ),
            ],
            coords,
            self.zen_model.lp_model,
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_storage_technology_capex", constraints
        )
