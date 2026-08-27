from zen_garden.topology.generic_parameter import GenericParameter


class Distance(GenericParameter):
    """Distance between nodes."""

    name = "distance"
    indices = ("set_transport_technologies", "set_edges")
    doc = "Distance between nodes"
    unit_category = {"distance": 1}
