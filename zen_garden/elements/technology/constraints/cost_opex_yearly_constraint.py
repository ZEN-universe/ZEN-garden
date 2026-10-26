from zen_garden.elements.technology.expressions.annual_fixed_opex import AnnualFixedOpex
from zen_garden.elements.technology.expressions.annual_variable_opex import (
    AnnualVariableOpex,
)
from zen_garden.elements.technology.masks import CapacityAnyTypeMask
from zen_garden.elements.technology.variables.cost_opex_yearly import CostOpexYearly
from zen_garden.model.component_types.constraint import GenericConstraint


class CostOpexYearlyConstraint(GenericConstraint):
    """Cost Opex Yearly."""

    name = "constraint_cost_opex_yearly"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAnyTypeMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Yearly opex for a technology at a location in each year.

        Formulation:

        .. math::
            C^{\\mathrm{op,ann}}_{h,p,y} =
            \\sum_{t\\in\\mathcal{T}_y}\\Delta t_t C^{\\mathrm{op,var}}_{h,p,t}
            + \\kappa^{\\mathrm{op,fix}}_{h,p,y} K_{h,p,y}

        For storage technologies, the implementation additionally includes the
        corresponding energy-capacity term:

        .. math::
            \\kappa^{\\mathrm{op,fix,energy}}_{h,n,y}K^{\\mathrm{energy}}_{h,n,y}

        Notation:

        :math:`C^{\\mathrm{op,ann}}_{h,p,y}`: OPEX of operating technology :math:`h` at
        location :math:`p` in year :math:`y`
        :math:`\\Delta t_t`: duration of time step :math:`t`
        :math:`C^{\\mathrm{op,var}}_{h,p,t}`: variable OPEX of operating technology
        :math:`h` at location :math:`p` in time step :math:`t` of year :math:`y`
        :math:`\\kappa^{\\mathrm{op,fix}}_{h,p,y}`: specific fixed OPEX
        :math:`K_{h,p,y}`: installed capacity of technology :math:`h` at location
        :math:`p` in year :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        return model.variables[CostOpexYearly] == (
            model.expressions[AnnualVariableOpex] + model.expressions[AnnualFixedOpex]
        )
