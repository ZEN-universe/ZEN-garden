"""Change in cumulative emissions between modeled years."""

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_annual import (
    CarbonEmissionsAnnual,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_cumulative import (
    CarbonEmissionsCumulative,
)
from zen_garden.model.component_types.expression import GenericExpression


class CumulativeEmissionsChange(GenericExpression):
    """Account for annual emissions between cumulative-emissions states."""

    name = "cumulative_emissions_change"
    indices = ("set_years",)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct account for annual emissions between cumulative-emissions states.

        Summary:
            Account for annual emissions between cumulative-emissions states.

        Formulation:

        .. math::

            E^{cum}_y-E^{cum}_{y-1}-E^{annual}_{y-1}(\\Delta y-1)-E^{annual}_y

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
        cumulative = optimization_model.variables[CarbonEmissionsCumulative]
        annual = optimization_model.variables[CarbonEmissionsAnnual]
        return (
            cumulative
            - cumulative.shift(set_years=1)
            - annual.shift(set_years=1)
            * (optimization_model.parameters[IntervalBetweenYears] - 1)
            - annual
        )
