from zen_garden.model.component_types.parameter import GenericParameter


class AvailabilityImport(GenericParameter):
    """Parameter which specifies the availability of carrier import."""

    name = "availability_import"
    symbol = r"a^{\mathrm{imp}}_{c,n,t}"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = (
        "Parameter which specifies the maximum energy that can be imported "
        "from outside the system boundaries."
    )
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
