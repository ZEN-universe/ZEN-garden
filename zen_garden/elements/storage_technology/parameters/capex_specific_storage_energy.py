from zen_garden.model.component_types.parameter import GenericParameter


class CapexSpecificStorageEnergy(GenericParameter):
    """CAPEX per unit of added storage energy capacity."""

    name = "capex_specific_storage_energy"
    symbol = r"\kappa^{\mathrm{cap,E}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Specific CAPEX of storage energy capacity."
    unit_category = {"money": 1, "energy_quantity": -1}
