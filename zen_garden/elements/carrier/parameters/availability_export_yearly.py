from zen_garden.model.component_types.parameter import GenericParameter


class AvailabilityExportYearly(GenericParameter):
    """Parameter which specifies the yearly availability of carrier export."""

    name = "availability_export_yearly"
    symbol = r"a^{\mathrm{exp,yr}}_{c,n,y}"
    indices = ("set_carriers", "set_nodes", "set_years")
    doc = (
        "Parameter which specifies the maximum energy that can be exported to "
        "outside the system boundaries for the entire year."
    )
    unit_category = {"energy_quantity": 1}
