from zen_garden.topology.generic_variable import GenericVariable


class ChargeStorageBinary(GenericVariable):
    """Variable for charge storage binary."""

    name = "charge_storage_binary"
    indices = ("set_storage_technologies", "set_nodes", "set_time_steps_operation")
    doc = "Variable for charge binary for storage technology"
    unit_category = None
    binary = True

