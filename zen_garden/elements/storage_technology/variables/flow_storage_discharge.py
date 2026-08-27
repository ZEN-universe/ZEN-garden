from zen_garden.topology.generic_variable import GenericVariable


class FlowStorageDischarge(GenericVariable):
    """Variable for carrier flow out of storage technology."""

    name = "flow_storage_discharge"
    indices = ["set_storage_technologies", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for carrier flow out of storage technology on node i and time t"
    unit_category = {"energy_quantity": 1, "time": -1}
