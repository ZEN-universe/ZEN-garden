from zen_garden.topology.generic_constraint import GenericConstraint


class CarbonEmissionsAnnualLimitConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
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
            model_constructor.zen_model.variables["carbon_emissions_annual"]
            - model_constructor.zen_model.variables["carbon_emissions_annual_overshoot"]
        )
        rhs = model_constructor.zen_model.parameters.carbon_emissions_annual_limit
        constraints = lhs <= rhs

        model_constructor.zen_model.add_constraint(
            "constraint_carbon_emissions_annual_limit", constraints
        )
