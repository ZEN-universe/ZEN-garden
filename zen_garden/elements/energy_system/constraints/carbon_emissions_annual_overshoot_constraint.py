import numpy as np

from zen_garden.topology.generic_constraint import GenericConstraint


class CarbonEmissionsAnnualOvershootConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Enforce zero annual overshoot when exactly one disabling condition holds.

        Annual overshoot is fixed to zero when either its price is infinite or no
        finite annual emissions limit exists, provided only one condition holds.

        Formulation:

        .. math::
            \\text{if exactly one of } \\pi^{\\mathrm{CO_2,ann}} =\\infty
            \\text{ and } \\overline{m}_y=\\infty\\;\\forall y
            \\text{, then: } M_y^{\\mathrm{ann,over}} = 0

        Notation:

        :math:`M_y^{\\mathrm{ann,over}}`: overshoot of the annual carbon emissions limit
        of energy system
        :math:`\\pi^{\\mathrm{CO_2,ann}}`: carbon price for annual overshoot
        :math:`\\overline{m}_y`: annual carbon emissions limit
        """
        no_price = (
            model_constructor.zen_model.parameters.price_carbon_emissions_annual_overshoot
            == np.inf
        )
        no_limit = (
            model_constructor.zen_model.parameters.carbon_emissions_annual_limit
            == np.inf
        ).all()
        if (no_price or no_limit) and not (no_price and no_limit):
            lhs = model_constructor.zen_model.variables[
                "carbon_emissions_annual_overshoot"
            ]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = None

        model_constructor.zen_model.add_constraint(
            "constraint_carbon_emissions_annual_overshoot", constraints
        )
