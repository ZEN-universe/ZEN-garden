from zen_garden.topology.generic_parameter import GenericParameter


class CapexSpecificTransport(GenericParameter):
    """Capex per capacity unit."""

    name = "capex_specific_transport"
    indices = ("set_transport_technologies", "set_edges", "set_years")
    doc = "Capex per capacity unit"
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
