import numpy as np

from zen_garden.elements.energy_system.parameters.price_carbon_emissions_budget_overshoot import (  # noqa: E501
    PriceCarbonEmissionsBudgetOvershoot,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_budget_overshoot import (  # noqa: E501
    CarbonEmissionsBudgetOvershoot,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsBudgetOvershootConstraint(GenericConstraint):
    """Carbon Emissions Budget Overshoot."""

    name = "constraint_carbon_emissions_budget_overshoot"
    indices = ("set_years",)

    @classmethod
    def should_construct(cls, model_constructor):
        return (
            model_constructor.optimization_model.parameters[
                PriceCarbonEmissionsBudgetOvershoot
            ]
            == np.inf
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Enforces zero budget overshoot if price for budget overshoot is inf.

        ensures carbon emissions overshoot of carbon budget is zero when
        carbon emissions price for budget overshoot is inf.

        Formulation:

        .. math::
            \\text{if } \\pi^{\\mathrm{CO_2,bud}} =\\infty
            \\text{, then: }M_y^{\\mathrm{bud,over}} = 0

        Notation:

        :math:`M_y^{\\mathrm{bud,over}}`: overshoot carbon emissions of energy system at
        the end of the time horizon
        :math:`\\pi^{\\mathrm{CO_2,bud}}`: carbon price for budget overshoot

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        return (
            model_constructor.optimization_model.variables[
                CarbonEmissionsBudgetOvershoot
            ]
            == 0
        )
