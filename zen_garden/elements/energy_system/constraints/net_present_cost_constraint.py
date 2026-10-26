from zen_garden.elements.energy_system.expressions.discounted_annual_system_cost import (  # noqa: E501
    DiscountedAnnualSystemCost,
)
from zen_garden.elements.energy_system.variables.net_present_cost import NetPresentCost
from zen_garden.model.component_types.constraint import GenericConstraint


class NetPresentCostConstraint(GenericConstraint):
    """Net Present Cost."""

    name = "constraint_net_present_cost"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Discounts the annual capital flows to calculate the net present cost.

        Formulation:

        .. math::
            C^{\\mathrm{NPC}}_y = \\sum_{i=0}^{\\delta_y-1}
            \\left( \\dfrac{1}{1+r^{\\mathrm{disc}}} \\right)^{\\Delta y(y-y_0)+i}
            C^{\\mathrm{total}}_y

        where :math:`\\delta_y=\\Delta y` for ordinary planning periods and
        :math:`\\delta_y=1` at the end of the planning horizon.

        Notation:

        :math:`\\delta_y=1` when :math:`y` is the final year of the entire horizon.

        :math:`C^{\\mathrm{NPC}}_y`: net present cost of energy system in year :math:`y`
        :math:`C^{\\mathrm{total}}_y`: total cost of energy system in year :math:`y`
        :math:`r^{\\mathrm{disc}}`: discount rate
        :math:`\\Delta y`: interval between planning periods

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        return (
            model.variables[NetPresentCost]
            == model.expressions[DiscountedAnnualSystemCost]
        )
