from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsBudgetConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
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
            year != model_constructor.model_schema.set_years_entire_horizon[-1]
            for year in model_constructor.model_schema.set_years
        ]

        lhs = (
            model_constructor.zen_model.variables["carbon_emissions_cumulative"]
            - model_constructor.zen_model.variables["carbon_emissions_budget_overshoot"]
            + (
                model_constructor.zen_model.variables["carbon_emissions_annual"].where(
                    m
                )
                * (model_constructor.config.system.interval_between_years - 1)
            )
        )
        rhs = model_constructor.zen_model.parameters.carbon_emissions_budget
        constraints = lhs <= rhs

        model_constructor.zen_model.add_constraint(
            "constraint_carbon_emissions_budget", constraints
        )
