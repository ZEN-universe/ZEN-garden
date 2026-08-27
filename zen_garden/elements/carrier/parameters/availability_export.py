from zen_garden.topology.generic_parameter import GenericParameter


class AvailabilityExport(GenericParameter):
    """Parameter which specifies the availability of carrier export."""

    name = "availability_export"
    indices = ("set_carriers", "set_nodes", "set_hours")
    doc = "Parameter which specifies the availability of carrier export"
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
