from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsBudgetConstraint(GenericConstraint):
    def build(self):
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
            year != self.energy_system.set_years_entire_horizon[-1]
            for year in self.energy_system.set_years
        ]

        lhs = (
            self.zen_model.variables["carbon_emissions_cumulative"]
            - self.zen_model.variables["carbon_emissions_budget_overshoot"]
            + (
                self.zen_model.variables["carbon_emissions_annual"].where(m)
                * (self.config.system.interval_between_years - 1)
            )
        )
        rhs = self.zen_model.parameters.carbon_emissions_budget
        constraints = lhs <= rhs

        self.zen_model.add_constraint("constraint_carbon_emissions_budget", constraints)
