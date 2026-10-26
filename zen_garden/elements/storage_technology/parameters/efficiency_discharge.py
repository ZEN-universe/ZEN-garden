from zen_garden.model.component_types.parameter import GenericParameter


class EfficiencyDischarge(GenericParameter):
    """Efficiency during discharging."""

    name = "efficiency_discharge"
    symbol = r"\eta^{\mathrm{dis}}_{h,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Efficiency during discharging for storage technologies."
    unit_category = {}
