from zen_garden.model.component_types.parameter import GenericParameter


class CapacityExistingEnergy(GenericParameter):
    """Existing storage energy capacity by vintage."""

    name = "capacity_existing_energy"
    symbol = r"e^{\mathrm{ex}}_{h,n,v}"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_technologies_existing",
    )
    doc = "Existing storage energy capacity by node and vintage."
    unit_category = {"energy_quantity": 1}
