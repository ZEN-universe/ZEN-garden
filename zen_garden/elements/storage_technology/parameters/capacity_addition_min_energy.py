from zen_garden.model.component_types.parameter import GenericParameter


class CapacityAdditionMinEnergy(GenericParameter):
    """Minimum storage energy-capacity addition."""

    name = "capacity_addition_min_energy"
    symbol = r"\underline{\Delta e}_{h}"
    indices = ("set_storage_technologies",)
    doc = "Minimum energy-capacity addition for a storage technology."
    unit_category = {"energy_quantity": 1}
