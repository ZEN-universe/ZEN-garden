from zen_garden.elements.carrier.parameters.price_export import PriceExport
from zen_garden.elements.carrier.parameters.price_import import PriceImport
from zen_garden.elements.carrier.variables.cost_carrier import CostCarrier
from zen_garden.elements.carrier.variables.flow_export import FlowExport
from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.model.component_types.constraint import GenericConstraint


class CostCarrierConstraint(GenericConstraint):
    """Cost Carrier."""

    name = "constraint_cost_carrier"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Cost of importing and exporting carrier.

        Formulation:

        .. math::
            C^{\\mathrm{carrier}}_{c,n,t} = \\pi^{\\mathrm{imp}}_{c,n,t}
            F^{\\mathrm{imp}}_{c,n,t} - \\pi^{\\mathrm{exp}}_{c,n,t}
            F^{\\mathrm{exp}}_{c,n,t}

        Notation:

        :math:`\\pi^{\\mathrm{imp}}_{c,n,t}`: import price of carrier :math:`c` at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`\\pi^{\\mathrm{exp}}_{c,n,t}`: export price of carrier :math:`c` at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{imp}}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{exp}}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` in time step :math:`t` of year :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        ### formulate constraint
        lhs = (
            model_constructor.optimization_model.variables[CostCarrier]
            - model_constructor.optimization_model.parameters[PriceImport]
            * model_constructor.optimization_model.variables[FlowImport]
            + model_constructor.optimization_model.parameters[PriceExport]
            * model_constructor.optimization_model.variables[FlowExport]
        )
        rhs = 0
        constraints = lhs == rhs

        return constraints
