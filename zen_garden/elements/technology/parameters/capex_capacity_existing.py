from zen_garden.model.component_types.parameter import GenericParameter


class CapexCapacityExisting(GenericParameter):
    """Total outstanding capex of an existing technology."""

    name = "capex_capacity_existing"
    symbol = r"\kappa^{\mathrm{cap,ex}}_{h,p,v}"
    indices = (
        "set_technologies",
        "set_location",
        "set_technologies_existing",
    )
    doc = (
        "Parameter which specifies the total capex of an existing technology "
        "which still has to be paid."
    )
    unit_category = {"money": 1}
    dependencies = ()

    @classmethod
    def get_input_data(cls, element):
        """Return no direct input because finalization calculates this parameter."""
        return None
