from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsAnnualLimitConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Time dependent carbon emissions limit from technologies and carriers.

        Formulation:

        .. math::
            M_y-M_y^{\\mathrm{ann,over}}\\leq \\overline{m}_y

        Notation:

        :math:`M_y^{\\mathrm{ann,over}}`: permitted overshoot of the annual emissions
        limit
        """
        lhs = (
            self.zen_model.variables["carbon_emissions_annual"]
            - self.zen_model.variables["carbon_emissions_annual_overshoot"]
        )
        rhs = self.zen_model.parameters.carbon_emissions_annual_limit
        constraints = lhs <= rhs

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_annual_limit", constraints
        )
