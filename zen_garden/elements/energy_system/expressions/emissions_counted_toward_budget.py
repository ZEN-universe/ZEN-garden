"""Emissions quantity evaluated against the carbon budget."""

import xarray as xr

from zen_garden.elements.energy_system.expressions.carbon_emissions_annual import (
    CarbonEmissionsAnnual,
)
from zen_garden.elements.energy_system.expressions.carbon_emissions_cumulative import (
    CarbonEmissionsCumulative,
)
from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_budget_overshoot import (  # noqa: E501
    CarbonEmissionsBudgetOvershoot,
)
from zen_garden.model.component_types.expression import GenericExpression


class EmissionsCountedTowardBudget(GenericExpression):
    """Combine cumulative, annual, and budget-overshoot emissions."""

    name = "emissions_counted_toward_budget"
    unit_category = {"emissions": 1}
    indices = ("set_years",)
    dependencies = (CarbonEmissionsCumulative, CarbonEmissionsAnnual)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct combine cumulative, annual, and budget-overshoot emissions.

        Summary:
            Combine cumulative, annual, and budget-overshoot emissions.

        Formulation:

        .. math::

            E^{cum}_y-E^{budget+}_y+\\mathbb1_{y\\ne y^{last}}E^{annual}_y(\\Delta y-1)

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
        years = model_constructor.model_schema.set_years
        not_last_year = xr.DataArray(
            [
                year != model_constructor.model_schema.set_years_entire_horizon[-1]
                for year in years
            ],
            coords={"set_years": years},
            dims="set_years",
        )
        return (
            optimization_model.expressions[CarbonEmissionsCumulative]
            - optimization_model.variables[CarbonEmissionsBudgetOvershoot]
            + optimization_model.expressions[CarbonEmissionsAnnual].where(not_last_year)
            * (optimization_model.parameters[IntervalBetweenYears] - 1)
        )
