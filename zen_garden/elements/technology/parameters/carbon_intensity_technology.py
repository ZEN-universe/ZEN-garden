from zen_garden.elements.transport_technology.parameters.distance import Distance
from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class CarbonIntensityTechnology(GenericParameter):
    """Carbon intensity of each technology."""

    name = "carbon_intensity_technology"
    symbol = r"\varepsilon^{\mathrm{op}}_{h}"
    indices = ("set_technologies", "set_location")
    doc = "Parameter which specifies the carbon intensity of each technology."
    unit_category = {"emissions": 1, "energy_quantity": -1}
    dependencies = (Distance,)

    @classmethod
    def get_input_data(cls, element):
        """Load carbon intensity after all transport distances are available."""
        data = super().get_input_data(element)
        assert data is not None
        unit = data.units["unit_in_base_units"].units
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
            return ParameterInputData(value * element.distance, element.units[cls.name])
        return data
