from zen_garden.model.component_types.expression import GenericExpression


class TotalCarbonEmissions(GenericExpression):
    """Total carbon emissions objective expression.

    .. math::
        J = E^{\\mathrm{cum}}_Y

    :math:`E^{\\mathrm{cum}}_Y`: cumulative carbon emissions at the end of the
    time horizon.
    """

    name = "total_carbon_emissions"
    doc = "Cumulative carbon emissions at the end of the time horizon"

    @classmethod
    def get_expression(cls, model_constructor):
        zen_model = model_constructor.zen_model
        return (
            zen_model.variables["carbon_emissions_cumulative"]
            .at[zen_model.sets["set_years"][-1]]
            .to_linexpr()
        )
