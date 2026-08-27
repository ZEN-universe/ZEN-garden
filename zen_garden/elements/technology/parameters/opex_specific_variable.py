from zen_garden.topology.generic_parameter import GenericParameter


class OpexSpecificVariable(GenericParameter):
    """Variable specific opex."""

    name = "opex_specific_variable"
    indices = ("set_technologies", "set_location", "set_hours")
    doc = "Variable specific opex"
    unit_category = {"money": 1, "energy_quantity": -1}
    time_series = True
