from zen_garden.model.component_types.parameter import GenericParameter


class CarbonIntensityCarrierExport(GenericParameter):
    """Parameter which specifies the carbon intensity of carrier export."""

    name = "carbon_intensity_carrier_export"
    indices = ("set_carriers", "set_nodes", "set_years")
    doc = "Parameter which specifies the carbon intensity of carrier export"
    unit_category = {"emissions": 1, "energy_quantity": -1}
