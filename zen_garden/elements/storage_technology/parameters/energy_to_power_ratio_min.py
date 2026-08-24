from zen_garden.topology.generic_parameter import GenericParameter


class EnergyToPowerRatioMin(GenericParameter):
    """Energy-to-power ratio lower bound."""

    name = "energy_to_power_ratio_min"
    indices = ("set_storage_technologies",)
    doc = "Energy-to-power ratio lower bound"
    unit_category = {}
