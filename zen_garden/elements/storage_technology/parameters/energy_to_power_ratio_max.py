from zen_garden.topology.generic_parameter import GenericParameter


class EnergyToPowerRatioMax(GenericParameter):
    """Energy-to-power ratio upper bound."""

    name = "energy_to_power_ratio_max"
    indices = ("set_storage_technologies",)
    doc = "Energy-to-power ratio upper bound"
    unit_category = {}
