from zen_garden.model.component_types.parameter import GenericParameter


class CapacityAdditionMaxEnergy(GenericParameter):
    """Maximum storage energy-capacity addition."""

    name = "capacity_addition_max_energy"
    symbol = r"\overline{\Delta e}_{h}"
    indices = ("set_storage_technologies",)
    doc = "Maximum energy-capacity addition for a storage technology."
    unit_category = {"energy_quantity": 1}
