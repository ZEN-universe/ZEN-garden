from zen_garden.model.component_types.parameter import GenericParameter


class CapacityInvestmentExisting(GenericParameter):
    """Parameter specifying the size of the previously invested capacities."""

    name = "capacity_investment_existing"
    indices = (
        "set_technologies",
        "set_capacity_types",
        "set_location",
        "set_years_entire_horizon",
    )
    doc = "Parameter specifying the size of the previously invested capacities"
    unit_category = {"energy_quantity": 1, "time": -1}
    capacity_types = True
