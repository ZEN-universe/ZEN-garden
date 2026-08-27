from zen_garden.topology.generic_parameter import GenericParameter


class SelfDischarge(GenericParameter):
    """Self-discharge of storage technologies."""

    name = "self_discharge"
    indices = ("set_storage_technologies", "set_nodes")
    doc = "Self-discharge of storage technologies"
    unit_category = {}
