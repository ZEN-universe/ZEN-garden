from zen_garden.model.component_types.parameter import GenericParameter


class CapexCapacityExistingEnergy(GenericParameter):
    """Outstanding CAPEX of existing storage energy capacity."""

    name = "capex_capacity_existing_energy"
    symbol = r"\kappa^{\mathrm{cap,ex,E}}_{h,n,v}"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_technologies_existing",
    )
    doc = "Outstanding CAPEX of existing storage energy capacity."
    unit_category = {"money": 1}

    @classmethod
    def get_input_data(cls, element):
        """Return no input because technology finalization calculates this value."""
        return None
