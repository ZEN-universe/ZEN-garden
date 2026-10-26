from zen_garden.model.component_types.parameter import GenericParameter


class Distance(GenericParameter):
    """Distance between nodes."""

    name = "distance"
    symbol = r"d^{\mathrm{dist}}_{h,e}"
    indices = ("set_transport_technologies", "set_edges")
    doc = "Distance between two nodes for transport technologies."
    unit_category = {"distance": 1}
