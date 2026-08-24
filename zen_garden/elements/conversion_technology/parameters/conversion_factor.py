from zen_garden.topology.generic_parameter import GenericParameter


class ConversionFactor(GenericParameter):
    """Conversion factor."""

    name = "conversion_factor"
    indices = (
        "set_conversion_technologies",
        "set_dependent_carriers",
        "set_nodes",
        "set_hours",
    )
    doc = "Conversion factor"
    unit_category = {}
    time_series = True
