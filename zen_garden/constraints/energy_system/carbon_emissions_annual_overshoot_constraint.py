import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsAnnualOvershootConstraint(GenericConstraint):
    def build(self):
        """Enforces zero annual overshoot if price for annual overshoot is inf.

        ensures annual carbon emissions overshoot is zero when carbon
        emissions price for annual overshoot is inf.

        .. math::
            \\text{if } \\mu^o =\\infty \\text{,then: } E_y^\\mathrm{o} = 0

        :math:`E_y^\\mathrm{o}`: overshoot of the annual carbon emissions limit
        of energy system \n
        :math:`\\mu^o`: carbon price for annual overshoot

        """
        no_price = (
            self.zen_model.parameters.price_carbon_emissions_annual_overshoot == np.inf
        )
        no_limit = (
            self.zen_model.parameters.carbon_emissions_annual_limit == np.inf
        ).all()
        if (no_price or no_limit) and not (no_price and no_limit):
            lhs = self.zen_model.variables["carbon_emissions_annual_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = None

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_annual_overshoot", constraints
        )
