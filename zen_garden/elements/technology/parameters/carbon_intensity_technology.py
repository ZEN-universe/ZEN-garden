from zen_garden.topology.generic_parameter import GenericComputedParameters


class CarbonIntensityTechnology(GenericComputedParameters):
    """Carbon intensity of each technology."""

    name = "carbon_intensity_technology"
    indices = ("set_technologies", "set_location")
    doc = "Carbon intensity of each technology"
    unit_category = {"emissions": 1, "energy_quantity": -1}
    input_loader = "carbon_intensity"
    dependencies = []
