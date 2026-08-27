from zen_garden.topology.generic_parameter import GenericParameter


class AvailabilityImportYearly(GenericParameter):
    """Parameter which specifies the yearly availability of carrier import."""

    name = "availability_import_yearly"
    indices = ("set_carriers", "set_nodes", "set_years")
    doc = "Parameter which specifies the yearly availability of carrier import"
    unit_category = {"energy_quantity": 1}
