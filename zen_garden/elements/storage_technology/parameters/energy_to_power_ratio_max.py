from zen_garden.model.component_types.parameter import GenericParameter


class EnergyToPowerRatioMax(GenericParameter):
    """Energy-to-power ratio upper bound."""

    name = "energy_to_power_ratio_max"
    symbol = r"r^{\mathrm{EP,max}}_{h}"
    indices = ("set_storage_technologies",)
    doc = "Maximum energy-to-power ratio for storage technologies."
    unit_category = {"time": 1}
