from zen_garden.model.component_types.parameter import GenericParameter


class AvailabilityImport(GenericParameter):
    """Parameter which specifies the availability of carrier import."""

    name = "availability_import"
    indices = ("set_carriers", "set_nodes", "set_hours")
    doc = "Parameter which specifies the availability of carrier import"
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
