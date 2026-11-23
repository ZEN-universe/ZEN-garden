from typing import Literal

from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.model.component_types.set import GenericRelationSet


def _edge_endpoint_combinations(model_constructor, endpoint: Literal[0, 1]):
    """Return one validated endpoint for every directed edge."""
    endpoints_by_edge = model_constructor.network_topology.set_nodes_on_edges
    combinations = []
    for edge in model_constructor.optimization_model.sets[SetEdges]:
        if edge not in endpoints_by_edge:
            raise ValueError(f"Edge {edge!r} has no configured endpoints")
        endpoints = endpoints_by_edge[edge]
        if len(endpoints) != 2:
            raise ValueError(
                f"Edge {edge!r} must have exactly two endpoints, "
                f"found {len(endpoints)}"
            )
        combinations.append((edge, endpoints[endpoint]))
    return combinations


class RelationEdgeStartNode(GenericRelationSet):
    """Start node of every directed edge."""

    name = "relation_edge_start_node"
    dimensions = (SetEdges, SetNodes)
    apply_by_default = False

    @classmethod
    def get_combinations(cls, model_constructor):
        return _edge_endpoint_combinations(model_constructor, 0)


class RelationEdgeEndNode(GenericRelationSet):
    """End node of every directed edge."""

    name = "relation_edge_end_node"
    dimensions = (SetEdges, SetNodes)
    apply_by_default = False

    @classmethod
    def get_combinations(cls, model_constructor):
        return _edge_endpoint_combinations(model_constructor, 1)
