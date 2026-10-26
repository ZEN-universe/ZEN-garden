from zen_garden.model.component_types.parameter import GenericParameter


class Demand(GenericParameter):
    """Parameter which specifies the carrier demand."""

    name = "demand"
    symbol = r"d_{c,n,t}"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = "Parameter which specifies the carrier demand."
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
