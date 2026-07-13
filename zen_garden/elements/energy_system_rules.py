import logging

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.elements.generic_rule import GenericRule

logger = logging.getLogger(__name__)


class EnergySystemRules(GenericRule):
    """This class takes care of the rules for the EnergySystem."""

    def constraint_carbon_emissions_cumulative(self):
        """Cumulative carbon emissions over time.

        .. math::
            \\text{First planning period } y = y_0, \\quad E_y^\\mathrm{cum} = E_y
        .. math::
            \\text{Subsequent periods } y > y_0, \\quad E_y^{cum}
            = E_{y-1}^{cum} + (dy-1)E_{y-1}+E_y

        :math:`dy`: interval between planning periods \n
        :math:`E_y`: annual carbon emissions in year :math:`y` \n
        :math:`E_y^{cum}`: cumulative carbon emissions in year :math:`y`

        """
        m = [
            True if year == self.energy_system.set_time_steps_yearly[0] else False
            for year in self.energy_system.set_time_steps_yearly
        ]

        lhs = (
            self.zen_model.lp_model.variables["carbon_emissions_cumulative"]
            - self.zen_model.lp_model.variables["carbon_emissions_cumulative"].shift(
                set_time_steps_yearly=1
            )
            - self.zen_model.lp_model.variables["carbon_emissions_annual"].shift(
                set_time_steps_yearly=1
            )
            * (self.config.system.interval_between_years - 1)
            - self.zen_model.lp_model.variables["carbon_emissions_annual"]
        )
        rhs = (
            xr.ones_like(
                self.zen_model.lp_model.variables["carbon_emissions_cumulative"].mask
            )
            * self.zen_model.parameters.carbon_emissions_cumulative_existing
        ).where(m, 0)
        constraints = lhs == rhs

        self.zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_cumulative", constraints
        )

    def constraint_carbon_emissions_annual_limit(self):
        """Time dependent carbon emissions limit from technologies and carriers.

        .. math::
            E_y\\leq e_y

        """
        lhs = (
            self.zen_model.lp_model.variables["carbon_emissions_annual"]
            - self.zen_model.lp_model.variables["carbon_emissions_annual_overshoot"]
        )
        rhs = self.zen_model.parameters.carbon_emissions_annual_limit
        constraints = lhs <= rhs

        self.zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_annual_limit", constraints
        )

    def constraint_carbon_emissions_budget(self):
        """Carbon emissions budget of whole time horizon.
        The prediction extends until the end of the horizon, i.e., last optimization
        time step plus the current carbon emissions until the end of the horizon.

        .. math::
            E_y^\\mathrm{cum} + (dy-1)  E_y - E_y^\\mathrm{bo} \\leq e^b

        :math:`E_y^\\mathrm{cum}`: cumulative carbon emissions of energy
        system in year :math:`y` \n
        :math:`E_y`: annual carbon emissions of energy system in year :math:`y` \n
        :math:`E_y^\\mathrm{bo}`: cumulative carbon emissions budget overshoot
        of energy system \n
        :math:`e^b`: carbon emissions budget of energy system

        """
        m = [
            year != self.energy_system.set_time_steps_yearly_entire_horizon[-1]
            for year in self.energy_system.set_time_steps_yearly
        ]

        lhs = (
            self.zen_model.lp_model.variables["carbon_emissions_cumulative"]
            - self.zen_model.lp_model.variables["carbon_emissions_budget_overshoot"]
            + (
                self.zen_model.lp_model.variables["carbon_emissions_annual"].where(m)
                * (self.config.system.interval_between_years - 1)
            )
        )
        rhs = self.zen_model.parameters.carbon_emissions_budget
        constraints = lhs <= rhs

        self.zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_budget", constraints
        )

    def constraint_net_present_cost(self):
        """Discounts the annual capital flows to calculate the net_present_cost.

        .. math::
            NPC_y = \\sum_{i \\in [0,dy(y))-1]}
            \\left( \\dfrac{1}{1+r} \\right)^{\\left(dy (y-y_0) + i \\right)} C_y

        :math:`NPC_y`: net present cost of energy system in year :math:`y` \n
        :math:`C_y`: total cost of energy system in year :math:`y` \n
        :math:`r`: discount rate \n
        :math:`dy`: interval between planning periods \n

        """
        factor = pd.Series(index=self.energy_system.set_time_steps_yearly)
        for year in self.energy_system.set_time_steps_yearly:
            ### auxiliary calculations
            if year == self.energy_system.set_time_steps_yearly_entire_horizon[-1]:
                interval_between_years = 1
            else:
                interval_between_years = self.config.system.interval_between_years
            # economic discount
            factor[year] = sum(
                (
                    (1 / (1 + self.zen_model.parameters.discount_rate))
                    ** (
                        self.config.system.interval_between_years
                        * (year - self.energy_system.set_time_steps_yearly[0])
                        + _intermediate_time_step
                    )
                )
                for _intermediate_time_step in range(0, interval_between_years)
            )
        term_discounted_cost_total = (
            self.zen_model.lp_model.variables["cost_total"] * factor
        )

        lhs = (
            self.zen_model.lp_model.variables["net_present_cost"]
            - term_discounted_cost_total
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.constraints.add_constraint(
            "constraint_net_present_cost", constraints
        )

    def constraint_carbon_emissions_budget_overshoot(self):
        """Enforces zero budget overshoot if price for budget overshoot is inf.

        ensures carbon emissions overshoot of carbon budget is zero when
        carbon emissions price for budget overshoot is inf.

        .. math::
            \\text{if } \\mu^{bo} =\\infty \\text{,then: }E_y^\\mathrm{bo} = 0

        :math:`E_y^\\mathrm{bo}`: overshoot carbon emissions of energy system at
        the end of the time horizon \n
        :math:`\\mu^{bo}`: carbon price for budget overshoot


        """
        if self.zen_model.parameters.price_carbon_emissions_budget_overshoot == np.inf:
            lhs = self.zen_model.lp_model.variables["carbon_emissions_budget_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = None

        self.zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_budget_overshoot", constraints
        )

    def constraint_carbon_emissions_annual_overshoot(self):
        """Enforces zero annual overshoot if price for annual overshoot is inf.

        ensures annual carbon emissions overshoot is zero when carbon
        emissions price for annual overshoot is inf.

        .. math::
            \\text{if } \\mu^o =\\infty \\text{,then: } E_y^\\mathrm{o} = 0

        :math:`E_y^\\mathrm{o}`: overshoot of the annual carbon emissions limit
        of energy system \n
        :math:`\\mu^o`: carbon price for annual overshoot

        """
        no_price = (
            self.zen_model.parameters.price_carbon_emissions_annual_overshoot == np.inf
        )
        no_limit = (
            self.zen_model.parameters.carbon_emissions_annual_limit == np.inf
        ).all()
        if (no_price or no_limit) and not (no_price and no_limit):
            lhs = self.zen_model.lp_model.variables["carbon_emissions_annual_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = None

        self.zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_annual_overshoot", constraints
        )

    def constraint_carbon_emissions_annual(self):
        """Add up all carbon emissions from technologies and carriers.

        .. math::
            E_y = E_{y,\\mathcal{H}} + E_{y,\\mathcal{C}}

        :math:`E_{y,\\mathcal{H}}`: carbon emissions from technologies in
        year :math:`y` \n
        :math:`E_{y,\\mathcal{C}}`: carbon emissions from carriers in year :math

        """
        lhs = (
            self.zen_model.lp_model.variables["carbon_emissions_annual"]
            - self.zen_model.lp_model.variables["carbon_emissions_technology_total"]
            - self.zen_model.lp_model.variables["carbon_emissions_carrier_total"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_annual", constraints
        )

    def constraint_cost_carbon_emissions_total(self):
        """Carbon cost associated with the carbon emissions of the system in each year.

        .. math::
            OPEX_y^\\mathrm{c} = E_y\\mu + E_y^\\mathrm{o}\\mu^\\mathrm{o}

        :math:`OPEX_y^\\mathrm{c}`: cost of carbon emissions in year :math:`y` \n
        :math:`E_y`: annual carbon emissions of energy system in year :math:`y` \n
        :math:`\\mu`: carbon price \n
        :math:`E_y^\\mathrm{o}`: annual carbon emissions overshoot in year :math:`y` \n
        :math:`\\mu^\\mathrm{o}`: carbon price for annual overshoot

        """
        mask_last_year = [
            year == self.energy_system.set_time_steps_yearly[-1]
            for year in self.energy_system.set_time_steps_yearly
        ]

        lhs = (
            self.zen_model.lp_model.variables["cost_carbon_emissions_total"]
            - self.zen_model.lp_model.variables["carbon_emissions_annual"]
            * self.zen_model.parameters.price_carbon_emissions
        )
        # add cost for overshooting carbon emissions budget
        budget_overshoot = (
            self.zen_model.parameters.price_carbon_emissions_budget_overshoot
        )
        if budget_overshoot != np.inf:
            lhs -= (
                self.zen_model.lp_model.variables[
                    "carbon_emissions_budget_overshoot"
                ].where(mask_last_year)
                * budget_overshoot.item()
            )
        # add cost for overshooting annual carbon emissions limit
        annual_overshoot = (
            self.zen_model.parameters.price_carbon_emissions_annual_overshoot
        )
        if annual_overshoot != np.inf:
            lhs -= (
                self.zen_model.lp_model.variables["carbon_emissions_annual_overshoot"]
                * annual_overshoot.item()
            )

        rhs = 0
        constraints = lhs == rhs

        self.zen_model.constraints.add_constraint(
            "constraint_cost_carbon_emissions_total", constraints
        )

    def constraint_cost_total(self):
        """Add up all costs from technologies and carriers.

        .. math::
            C_y = CAPEX_y + OPEX_y^\\mathrm{t} + OPEX_y^\\mathrm{c} + OPEX_y^\\mathrm{e}

        :math:`C_y`: total cost of energy system in year :math:`y` \n
        :math:`CAPEX_y`: annual capital expenditures in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{t}`: annual operational expenditures for operating
        technologies in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{c}`: annual operational expenditures for for importing
        and exporting carriers in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{e}`: annual operational expenditures for carbon
        emissions in year :math:`y`

        """
        lhs = (
            self.zen_model.lp_model.variables["cost_total"]
            - self.zen_model.lp_model.variables["cost_capex_yearly_total"]
            - self.zen_model.lp_model.variables["cost_opex_yearly_total"]
            - self.zen_model.lp_model.variables["cost_carrier_total"]
            - self.zen_model.lp_model.variables["cost_carbon_emissions_total"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.constraints.add_constraint("constraint_cost_total", constraints)

    # Objective rules
    # ---------------

    def objective_total_cost(self):
        """Objective function to minimize the total net present cost.

        .. math::
            J = \\sum_{y\\in\\mathcal{Y}} NPC_y

        :param model: optimization model
        :return: net present cost objective function
        """
        return self.zen_model.lp_model.variables["net_present_cost"].sum(
            "set_time_steps_yearly"
        )

    def objective_total_carbon_emissions(self):
        """Objective function to minimize total emissions.

        .. math::
            J = E^{\\mathrm{cum}}_Y

        :math:`E^{\\mathrm{cum}}_Y`: cumulative carbon emissions at the end of
        the time horizon

        :param model: optimization model
        :return: total carbon emissions objective function
        """
        return (
            self.zen_model.lp_model.variables["carbon_emissions_cumulative"]
            .at[self.zen_model.sets["set_time_steps_yearly"][-1]]
            .to_linexpr()
        )
