from zen_garden.model.component_types.parameter import GenericParameter


class EfficiencyCharge(GenericParameter):
    """Efficiency during charging."""

    name = "efficiency_charge"
    symbol = r"\eta^{\mathrm{ch}}_{h,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Efficiency during charging for storage technologies."
    unit_category = {}
