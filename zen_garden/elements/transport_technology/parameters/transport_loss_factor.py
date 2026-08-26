from zen_garden.topology.generic_parameter import GenericComputedParameters


class TransportLossFactor(GenericComputedParameters):
    """Carrier losses due to transport."""

    name = "transport_loss_factor"
    indices = ("set_transport_technologies", "set_edges")
    doc = "Carrier losses due to transport"
    unit_category = {}
    input_loader = "transport_loss"
    dependencies = ["distance"]

    @classmethod
    def store_input_data(cls, element, loader):
        loader.load_into(cls, element)
