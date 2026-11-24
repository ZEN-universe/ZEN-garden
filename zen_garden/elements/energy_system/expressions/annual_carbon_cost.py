"""Annual carbon-emissions cost."""

import numpy as np

from zen_garden.elements.energy_system.expressions.carbon_emissions_annual import (
    CarbonEmissionsAnnual,
)
from zen_garden.elements.energy_system.parameters.price_carbon_emissions import (
    PriceCarbonEmissions,
)
from zen_garden.elements.energy_system.parameters.price_carbon_emissions_annual_overshoot import (  # noqa: E501
    PriceCarbonEmissionsAnnualOvershoot,
)
from zen_garden.elements.energy_system.parameters.price_carbon_emissions_budget_overshoot import (  # noqa: E501
    PriceCarbonEmissionsBudgetOvershoot,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_annual_overshoot import (  # noqa: E501
    CarbonEmissionsAnnualOvershoot,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_budget_overshoot import (  # noqa: E501
    CarbonEmissionsBudgetOvershoot,
)
from zen_garden.model.component_types.expression import GenericExpression


class AnnualCarbonCost(GenericExpression):
    """Combine emissions prices and configured overshoot penalties."""

    name = "annual_carbon_cost"
    unit_category = {"money": 1}
    indices = ("set_years",)
    dependencies = (CarbonEmissionsAnnual,)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct combine emissions prices and configured overshoot penalties.

        Summary:
            Combine emissions prices and configured overshoot penalties.

        Formulation:

        .. math::

            E^{annual}_yp^{CO2}_y +\\mathbb1_{y =y^{last}}E^{budget +}p^{budget +}
            +E^{annual +}_yp^{annual +}

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
        expression = (
            optimization_model.expressions[CarbonEmissionsAnnual]
            * optimization_model.parameters[PriceCarbonEmissions]
        )
        budget_price = optimization_model.parameters[
            PriceCarbonEmissionsBudgetOvershoot
        ]
        if budget_price != np.inf:
            last_year = [
                year == model_constructor.model_schema.set_years[-1]
                for year in model_constructor.model_schema.set_years
            ]
            expression += (
                optimization_model.variables[CarbonEmissionsBudgetOvershoot].where(
                    last_year
                )
                * budget_price.item()
            )
        annual_price = optimization_model.parameters[
            PriceCarbonEmissionsAnnualOvershoot
        ]
        if annual_price != np.inf:
            expression += (
                optimization_model.variables[CarbonEmissionsAnnualOvershoot]
                * annual_price.item()
            )
        return expression
