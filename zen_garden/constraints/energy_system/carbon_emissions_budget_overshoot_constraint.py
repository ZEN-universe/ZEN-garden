import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsBudgetOvershootConstraint(GenericConstraint):
    def build(self):
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
        """
        if self.zen_model.parameters.price_carbon_emissions_budget_overshoot == np.inf:
            lhs = self.zen_model.variables["carbon_emissions_budget_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = None

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_budget_overshoot", constraints
        )
