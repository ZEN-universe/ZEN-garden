from zen_garden.model.component_types.parameter import GenericParameter


class CapacityInvestmentExistingEnergy(GenericParameter):
    """Storage energy capacity invested before the modeled horizon."""

    name = "capacity_investment_existing_energy"
    symbol = r"e^{\mathrm{ex,inv}}_{h,n,v}"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years_entire_horizon",
    )
    doc = "Pre-horizon investments in storage energy capacity."
    unit_category = {"energy_quantity": 1}
