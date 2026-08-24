from zen_garden.topology.generic_parameter import GenericParameter


class FlowStorageInflow(GenericParameter):
    """Energy inflow into storage technologies."""

    name = "flow_storage_inflow"
    indices = ("set_storage_technologies", "set_nodes", "set_hours")
    doc = "Energy inflow into storage technologies"
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
