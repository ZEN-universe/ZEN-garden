from zen_garden.model.component_types.parameter import GenericParameter


class CapacityInvestmentExisting(GenericParameter):
    """Parameter specifying the size of the previously invested capacities."""

    name = "capacity_investment_existing"
    symbol = r"k^{\mathrm{ex,inv}}_{h,p,v}"
    indices = (
        "set_technologies",
        "set_location",
        "set_years_entire_horizon",
    )
    doc = "Parameter which specifies the size of the previously invested capacities."
    unit_category = {"energy_quantity": 1, "time": -1}
