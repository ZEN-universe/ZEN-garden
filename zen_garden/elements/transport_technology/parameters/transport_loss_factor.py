from zen_garden.topology.generic_parameter import GenericParameter


class TransportLossFactor(GenericParameter):
    """Carrier losses due to transport."""

    name = "transport_loss_factor"
    indices = ("set_transport_technologies", "set_edges")
    doc = "Carrier losses due to transport"
    unit_category = {}
