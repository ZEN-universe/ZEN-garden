from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.energy_system.variables.carbon_emissions_cumulative import (
    CarbonEmissionsCumulative,
)
from zen_garden.model.component_types.expression import GenericExpression


class TotalCarbonEmissions(GenericExpression):
    """Total carbon emissions objective expression.

    .. math::
        J = E^{\\mathrm{cum}}_Y

    :math:`E^{\\mathrm{cum}}_Y`: cumulative carbon emissions at the end of the
    time horizon.
    """

    name = "total_carbon_emissions"
    indices = ()
    doc = "Cumulative carbon emissions at the end of the time horizon"

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct total carbon emissions objective expression.

        Summary:
            Total carbon emissions objective expression.

        Formulation:

        .. math::

            E^{cum}_{y^{last}}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        return (
            optimization_model.variables[CarbonEmissionsCumulative]
            .at[optimization_model.sets[SetYears][-1]]
            .to_linexpr()
        )
