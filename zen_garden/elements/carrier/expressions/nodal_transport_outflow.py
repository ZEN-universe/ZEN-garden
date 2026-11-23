"""Transport outflow by carrier and node."""

from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.elements.energy_system.sets.relation_edge_node import (
    RelationEdgeStartNode,
)
from zen_garden.elements.energy_system.sets.set_carriers import SetCarriers
from zen_garden.elements.technology.sets.relation_technology_reference_carrier import (  # noqa: E501
    RelationTechnologyReferenceCarrier,
)
from zen_garden.elements.transport_technology.variables.flow_transport import (
    FlowTransport,
)
from zen_garden.model.component_types.expression import GenericExpression


class NodalTransportOutflow(GenericExpression):
    """Aggregate outgoing transport flow."""

    name = "nodal_transport_outflow"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct aggregate outgoing transport flow.

        Summary:
            Aggregate outgoing transport flow.

        Formulation:

        .. math::

            \\sum_{h,e\\in\\delta^+(n)}F^{trans}_{h,e,y,r}

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
        if optimization_model.variables[FlowTransport].size == 0:
            return optimization_model.variables[FlowImport].where(False).to_linexpr()

        sets = optimization_model.sets
        carrier_mask = sets.relation_mask(
            RelationTechnologyReferenceCarrier,
            dimensions=("set_transport_technologies", SetCarriers),
        )
        carrier_mask = carrier_mask.reindex(
            {SetCarriers.name: sets[SetCarriers].coordinate_values},
            fill_value=False,
        )
        start_node_mask = sets.relation_mask(RelationEdgeStartNode)
        valid_flow = carrier_mask & start_node_mask
        return (
            optimization_model.variables[FlowTransport]
            .where(valid_flow)
            .sum(["set_transport_technologies", "set_edges"])
        )
