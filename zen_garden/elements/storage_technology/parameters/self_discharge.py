from zen_garden.model.component_types.parameter import GenericParameter


class SelfDischarge(GenericParameter):
    """Self-discharge of storage technologies."""

    name = "self_discharge"
    symbol = r"\lambda^{\mathrm{self}}_{h}"
    indices = ("set_storage_technologies", "set_nodes")
    doc = "Self-discharge rate of storage technologies."
    unit_category = {}
