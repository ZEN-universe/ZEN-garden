from zen_garden.topology.generic_variable import GenericVariable


class FlowConversionInput(GenericVariable):
    """Variable for carrier input of conversion technologies."""

    name = "flow_conversion_input"
    indices = ("set_conversion_technologies", "set_input_carriers", "set_nodes", "set_time_steps_operation")
    doc = "Variable for carrier input of conversion technologies"
    unit_category = {"energy_quantity": 1, "time": -1}

