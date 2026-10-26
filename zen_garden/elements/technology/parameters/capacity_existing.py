from zen_garden.model.component_types.parameter import GenericParameter


class CapacityExisting(GenericParameter):
    """Parameter which specifies the existing technology size."""

    name = "capacity_existing"
    symbol = r"k^{\mathrm{ex}}_{h,p,v}"
    indices = (
        "set_technologies",
        "set_location",
        "set_technologies_existing",
    )
    doc = "Parameter which specifies the existing technology size."
    unit_category = {"energy_quantity": 1, "time": -1}
