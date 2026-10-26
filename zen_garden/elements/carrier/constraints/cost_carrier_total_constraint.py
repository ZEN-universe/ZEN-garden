from zen_garden.elements.carrier.expressions.annual_carrier_cost import (
    AnnualCarrierCost,
)
from zen_garden.elements.carrier.variables.cost_carrier_total import CostCarrierTotal
from zen_garden.model.component_types.constraint import GenericConstraint


class CostCarrierTotalConstraint(GenericConstraint):
    """Cost Carrier Total."""

    name = "constraint_cost_carrier_total"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Total cost of importing and exporting carrier.

        Formulation:

        .. math::
            C^{\\mathrm{carrier}}_y = \\sum_{c\\in\\mathcal{C}}\\sum_{n\\in\\mathcal{N}}
            \\sum_{t\\in\\mathcal{T}_y} \\Delta t_t (C^{\\mathrm{carrier}}_{c,n,t}
            + C^{\\mathrm{shed}}_{c,n,t})

        Notation:

        :math:`C^{\\mathrm{carrier}}_{c,n,t}`: cost of importing and exporting carrier
        :math:`c` at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`C^{\\mathrm{shed}}_{c,n,t}`: cost of shedding
        demand of carrier :math:`c` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        :math:`\\Delta t_t`: duration of time step :math:`t`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        return model.variables[CostCarrierTotal] == model.expressions[AnnualCarrierCost]
