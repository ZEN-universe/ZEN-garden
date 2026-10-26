from zen_garden.model.component_types.parameter import GenericParameter


class AvailabilityImportYearly(GenericParameter):
    """Parameter which specifies the yearly availability of carrier import."""

    name = "availability_import_yearly"
    symbol = r"a^{\mathrm{imp,yr}}_{c,n,y}"
    indices = ("set_carriers", "set_nodes", "set_years")
    doc = (
        "Parameter which specifies the maximum energy that can be imported "
        "from outside the system boundaries for the entire year."
    )
    unit_category = {"energy_quantity": 1}
