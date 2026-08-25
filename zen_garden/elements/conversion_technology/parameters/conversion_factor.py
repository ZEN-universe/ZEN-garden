from zen_garden.topology.generic_parameter import GenericComputedParameters


class ConversionFactor(GenericComputedParameters):
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
    input_loader = "dependent_carrier"
    input_indices = ("set_nodes", "set_hours")
    dependencies = []
