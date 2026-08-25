from zen_garden.topology.generic_parameter import GenericComputedParameters


class CapexSpecificStorage(GenericComputedParameters):
    """Specific capex of storage technologies."""

    name = "capex_specific_storage"
    indices = (
        "set_storage_technologies",
        "set_capacity_types",
        "set_nodes",
        "set_years",
    )
    doc = "Specific capex of storage technologies"
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
    capacity_types = True
    input_loader = "storage_capex"
    dependencies = []
