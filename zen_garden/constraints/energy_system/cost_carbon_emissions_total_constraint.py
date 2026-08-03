import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class CostCarbonEmissionsTotalConstraint(GenericConstraint):
    def build(self):
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
            year == self.energy_system.set_years[-1]
            for year in self.energy_system.set_years
        ]

        lhs = (
            self.zen_model.variables["cost_carbon_emissions_total"]
            - self.zen_model.variables["carbon_emissions_annual"]
            * self.zen_model.parameters.price_carbon_emissions
        )
        # add cost for overshooting carbon emissions budget
        budget_overshoot = (
            self.zen_model.parameters.price_carbon_emissions_budget_overshoot
        )
        if budget_overshoot != np.inf:
            lhs -= (
                self.zen_model.variables["carbon_emissions_budget_overshoot"].where(
                    mask_last_year
                )
                * budget_overshoot.item()
            )
        # add cost for overshooting annual carbon emissions limit
        annual_overshoot = (
            self.zen_model.parameters.price_carbon_emissions_annual_overshoot
        )
        if annual_overshoot != np.inf:
            lhs -= (
                self.zen_model.variables["carbon_emissions_annual_overshoot"]
                * annual_overshoot.item()
            )

        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_cost_carbon_emissions_total", constraints
        )
