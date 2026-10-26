from zen_garden.elements.storage_technology.variables.flow_storage_charge import (
    FlowStorageCharge,
)


class FlowStorageDischarge(FlowStorageCharge):
    """Variable for carrier flow out of storage technology."""

    name = "flow_storage_discharge"
    symbol = r"F^{\mathrm{dis}}_{h,n,t}"
    indices = [
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Carrier flow discharged from a storage technology."
    unit_category = {"energy_quantity": 1, "time": -1}
