from zen_garden.model.component_types.parameter import GenericParameter


class AvailabilityExport(GenericParameter):
    """Parameter which specifies the availability of carrier export."""

    name = "availability_export"
    symbol = r"a^{\mathrm{exp}}_{c,n,t}"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = (
        "Parameter which specifies the maximum energy that can be exported to "
        "outside the system boundaries."
    )
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
