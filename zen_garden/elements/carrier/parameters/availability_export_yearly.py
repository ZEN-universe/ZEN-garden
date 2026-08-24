from zen_garden.topology.generic_parameter import GenericParameter


class AvailabilityExportYearly(GenericParameter):
    """Parameter which specifies the yearly availability of carrier export."""

    name = "availability_export_yearly"
    indices = ("set_carriers", "set_nodes", "set_years")
    doc = "Parameter which specifies the yearly availability of carrier export"
    unit_category = {"energy_quantity": 1}
