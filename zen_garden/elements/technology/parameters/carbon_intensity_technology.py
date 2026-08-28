from zen_garden.topology.generic_parameter import GenericParameter


class CarbonIntensityTechnology(GenericParameter):
    """Carbon intensity of each technology."""

    name = "carbon_intensity_technology"
    indices = ("set_technologies", "set_location")
    doc = "Carbon intensity of each technology"
    unit_category = {"emissions": 1, "energy_quantity": -1}
    dependencies = ["distance"]

    @classmethod
    def store_input_data(cls, element):
        """Load carbon intensity after all transport distances are available."""
        super().store_input_data(element)
        unit = element.units[cls.name]["unit_in_base_units"].units
        if getattr(element, "location_type", None) == "set_edges" and (
            "/ kilometer" in str(unit)
        ):
            value = element.element_data_loader.extract_input_data(
                cls.name,
                index_sets=["set_edges"],
                unit_category={
                    "emissions": 1,
                    "energy_quantity": -1,
                    "distance": -1,
                },
            )
            cls._store_value(element, cls.name, value * element.distance)
