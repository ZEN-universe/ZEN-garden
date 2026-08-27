from zen_garden.topology.generic_parameter import GenericParameter


class PriceExport(GenericParameter):
    """Parameter which specifies the price of carrier export."""

    name = "price_export"
    indices = ("set_carriers", "set_nodes", "set_hours")
    doc = "Parameter which specifies the price of carrier export"
    unit_category = {"money": 1, "energy_quantity": -1}
    time_series = True
