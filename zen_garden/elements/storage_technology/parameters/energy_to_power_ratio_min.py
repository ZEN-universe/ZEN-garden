from zen_garden.model.component_types.parameter import GenericParameter


class EnergyToPowerRatioMin(GenericParameter):
    """Energy-to-power ratio lower bound."""

    name = "energy_to_power_ratio_min"
    symbol = r"r^{\mathrm{EP,min}}_{h}"
    indices = ("set_storage_technologies",)
    doc = "Minimum energy-to-power ratio for storage technologies."
    unit_category = {"time": 1}
