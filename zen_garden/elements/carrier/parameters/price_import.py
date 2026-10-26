from zen_garden.model.component_types.parameter import GenericParameter


class PriceImport(GenericParameter):
    """Parameter which specifies the price of carrier import."""

    name = "price_import"
    symbol = r"\pi^{\mathrm{imp}}_{c,n,t}"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = "Parameter which specifies the import carrier price."
    unit_category = {"money": 1, "energy_quantity": -1}
    time_series = True
