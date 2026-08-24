from zen_garden.topology.generic_parameter import GenericParameter


class CarbonIntensityTechnology(GenericParameter):
    """Carbon intensity of each technology."""

    name = "carbon_intensity_technology"
    indices = ("set_technologies", "set_location")
    doc = "Carbon intensity of each technology"
    unit_category = {"emissions": 1, "energy_quantity": -1}
