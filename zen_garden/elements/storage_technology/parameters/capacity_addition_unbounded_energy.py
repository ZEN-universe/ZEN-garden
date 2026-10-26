from zen_garden.model.component_types.parameter import GenericParameter


class CapacityAdditionUnboundedEnergy(GenericParameter):
    """Storage energy capacity exempt from diffusion limits."""

    name = "capacity_addition_unbounded_energy"
    symbol = r"e^{\mathrm{add,free}}_{h}"
    indices = ("set_storage_technologies",)
    doc = "Energy-capacity addition available without prior market knowledge."
    unit_category = {"energy_quantity": 1}
