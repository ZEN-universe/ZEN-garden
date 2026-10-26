from zen_garden.model.component_types.parameter import GenericParameter


class PriceExport(GenericParameter):
    """Parameter which specifies the price of carrier export."""

    name = "price_export"
    symbol = r"\pi^{\mathrm{exp}}_{c,n,t}"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = "Parameter which specifies the export carrier price."
    unit_category = {"money": 1, "energy_quantity": -1}
    time_series = True
