from zen_garden.topology.generic_constraint import GenericConstraint


class CarbonEmissionsBudgetConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Carbon emissions budget of whole time horizon.
        The prediction extends until the end of the horizon, i.e., last optimization
        time step plus the current carbon emissions until the end of the horizon.

        Formulation:

        .. math::
            M_y^{\\mathrm{cum}}
            + \\mathbb{1}_{y\\neq y_\\mathrm{H}}(\\Delta y-1)M_y
            - M_y^{\\mathrm{bud,over}} \\leq \\overline{m}^{\\mathrm{budget}}

        Notation:

        :math:`M_y^{\\mathrm{cum}}`: cumulative carbon emissions of energy
        system in year :math:`y`
        :math:`M_y`: annual carbon emissions of energy system in year :math:`y`
        :math:`M_y^{\\mathrm{bud,over}}`: cumulative carbon emissions budget overshoot
        of energy system
        :math:`\\overline{m}^{\\mathrm{budget}}`: carbon emissions budget of energy
        system
        :math:`y_\\mathrm{H}`: final year of the entire optimization horizon. The
        extrapolation term is omitted there because no intermediate years remain.
        """
        m = [
            year != self.model_schema.set_years_entire_horizon[-1]
            for year in self.model_schema.set_years
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
