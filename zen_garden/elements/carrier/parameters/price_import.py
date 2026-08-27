from zen_garden.topology.generic_parameter import GenericParameter


class PriceImport(GenericParameter):
    """Parameter which specifies the price of carrier import."""

    name = "price_import"
    indices = ("set_carriers", "set_nodes", "set_hours")
    doc = "Parameter which specifies the price of carrier import"
    unit_category = {"money": 1, "energy_quantity": -1}
    time_series = True
