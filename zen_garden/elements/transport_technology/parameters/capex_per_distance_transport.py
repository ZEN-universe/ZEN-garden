from zen_garden.topology.generic_parameter import GenericComputedParameters


class CapexPerDistanceTransport(GenericComputedParameters):
    """Capex per distance."""

    name = "capex_per_distance_transport"
    indices = ("set_transport_technologies", "set_edges", "set_years")
    doc = "Capex per distance"
    unit_category = {"money": 1, "distance": -1}
    input_loader = "skip"
    dependencies = ["capex_specific_transport"]

    @classmethod
    def store_input_data(cls, element, loader):
        loader.load_into(cls, element)
