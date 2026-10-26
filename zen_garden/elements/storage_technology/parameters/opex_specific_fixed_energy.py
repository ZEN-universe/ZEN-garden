from zen_garden.model.component_types.parameter import GenericParameter


class OpexSpecificFixedEnergy(GenericParameter):
    """Fixed annual OPEX per unit of storage energy capacity."""

    name = "opex_specific_fixed_energy"
    symbol = r"\kappa^{\mathrm{op,fix,E}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Fixed annual OPEX per unit of installed storage energy capacity."
    unit_category = {"money": 1, "energy_quantity": -1}
