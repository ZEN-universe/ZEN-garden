from zen_garden.topology.generic_parameter import GenericParameter


class CapexCapacityExisting(GenericParameter):
    """Total outstanding capex of an existing technology."""

    name = "capex_capacity_existing"
    indices = (
        "set_technologies",
        "set_capacity_types",
        "set_location",
        "set_technologies_existing",
    )
    doc = "Total outstanding capex of an existing technology"
    unit_category = {"money": 1}
    capacity_types = True
    input_loader = "skip"
