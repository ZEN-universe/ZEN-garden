from zen_garden.topology.generic_parameter import GenericComputedParameters


class CapexSpecificTransport(GenericComputedParameters):
    """Capex per capacity unit."""

    name = "capex_specific_transport"
    indices = ("set_transport_technologies", "set_edges", "set_years")
    doc = "Capex per capacity unit"
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
    input_loader = "transport_capex"
    dependencies = ["distance"]

    @classmethod
    def store_input_data(cls, element, loader):
        loader.load_into(cls, element)
