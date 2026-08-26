from zen_garden.topology.generic_variable import GenericVariable


class FlowConversionOutput(GenericVariable):
    """Variable for carrier output of conversion technologies."""

    name = "flow_conversion_output"
    indices = ("set_conversion_technologies", "set_output_carriers", "set_nodes", "set_time_steps_operation")
    doc = "Variable for carrier output of conversion technologies"
    unit_category = {"energy_quantity": 1, "time": -1}

