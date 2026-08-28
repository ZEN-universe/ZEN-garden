from zen_garden.model.component_types.parameter import GenericParameter


class EfficiencyDischarge(GenericParameter):
    """Efficiency during discharging."""

    name = "efficiency_discharge"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Efficiency during discharging"
    unit_category = {}
