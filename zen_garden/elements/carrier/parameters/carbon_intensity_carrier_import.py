from zen_garden.topology.generic_parameter import GenericParameter


class CarbonIntensityCarrierImport(GenericParameter):
    """Parameter which specifies the carbon intensity of carrier import."""

    name = "carbon_intensity_carrier_import"
    indices = ("set_carriers", "set_nodes", "set_years")
    doc = "Parameter which specifies the carbon intensity of carrier import"
    unit_category = {"emissions": 1, "energy_quantity": -1}
