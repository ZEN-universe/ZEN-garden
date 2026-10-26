from zen_garden.model.component_types.parameter import GenericParameter


class FlowStorageInflow(GenericParameter):
    """Energy inflow into storage technologies."""

    name = "flow_storage_inflow"
    symbol = r"q^{\mathrm{in}}_{h,n,t}"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = "Energy inflow for storage technology."
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
