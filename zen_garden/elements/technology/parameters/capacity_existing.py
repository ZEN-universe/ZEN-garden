from zen_garden.topology.generic_parameter import GenericParameter


class CapacityExisting(GenericParameter):
    """Parameter which specifies the existing technology size."""

    name = "capacity_existing"
    indices = (
        "set_technologies",
        "set_capacity_types",
        "set_location",
        "set_technologies_existing",
    )
    doc = "Parameter which specifies the existing technology size"
    unit_category = {"energy_quantity": 1, "time": -1}
    capacity_types = True
