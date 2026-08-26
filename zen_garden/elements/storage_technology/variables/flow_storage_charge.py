from zen_garden.topology.generic_variable import GenericVariable


class FlowStorageCharge(GenericVariable):
    """Variable for carrier flow into storage technology."""

    name = "flow_storage_charge"
    indices = ["set_storage_technologies", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for carrier flow into storage technology on node i and time t"
    unit_category = {"energy_quantity": 1, "time": -1}

