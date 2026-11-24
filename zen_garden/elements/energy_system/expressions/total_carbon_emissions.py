from zen_garden.elements.energy_system.expressions.carbon_emissions_cumulative import (
    CarbonEmissionsCumulative,
)
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.model.component_types.expression import GenericExpression


class TotalCarbonEmissions(GenericExpression):
    """Cumulative carbon emissions selected at the final modeled year."""

    name = "total_carbon_emissions"
    unit_category = {"emissions": 1}
    indices = ()
    doc = (
        "Cumulative carbon emissions at the final modeled year. The complete "
        "term selected at that year is expanded below.\n\n"
        + CarbonEmissionsCumulative.doc
    )
    dependencies = (CarbonEmissionsCumulative,)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct total carbon emissions objective expression.

        Summary:
            Total carbon emissions objective expression.

        The complete mathematical term is expanded in :attr:`doc`.

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
        return optimization_model.expressions[CarbonEmissionsCumulative].sel(
            set_years=optimization_model.sets[SetYears][-1]
        )
