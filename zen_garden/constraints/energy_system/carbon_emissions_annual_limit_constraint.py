from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsAnnualLimitConstraint(GenericConstraint):
    def build(self):
        """Time dependent carbon emissions limit from technologies and carriers.

        .. math::
            E_y\\leq e_y

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
