from zen_garden.topology.generic_variable import GenericVariable


class FlowTransport(GenericVariable):
    """Variable for carrier flow through transport technology."""

    name = "flow_transport"
    indices = ["set_transport_technologies", "set_edges", "set_time_steps_operation"]
    doc = "Variable for carrier flow through transport technology on edge i and time t"
    unit_category = {"energy_quantity": 1, "time": -1}
