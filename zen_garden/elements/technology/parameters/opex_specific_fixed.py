from zen_garden.elements.transport_technology.parameters.distance import Distance
from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class OpexSpecificFixed(GenericParameter):
    """Fixed annual specific opex."""

    name = "opex_specific_fixed"
    symbol = r"\kappa^{\mathrm{op,fix}}_{h,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Parameter which specifies the fixed annual specific opex."
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
    dependencies = (Distance,)

    @classmethod
    def get_input_data(cls, element):
        """Load fixed opex, including transport inputs specified per distance."""
        if getattr(element, "location_type", None) != "set_edges":
            return super().get_input_data(element)

        attributes = element.element_data_loader.attribute_dict
        indices = ["set_edges", "set_years"]
        if "opex_specific_fixed_per_distance" in attributes:
            per_distance = element.element_data_loader.extract_input_data(
                "opex_specific_fixed_per_distance",
                indices,
                {
                    "money": 1,
                    "distance": -1,
                    "energy_quantity": -1,
                    "time": 1,
                },
            )
            value = per_distance * element.distance
        elif cls.name in attributes:
            value = element.element_data_loader.extract_input_data(
                cls.name, indices, cls.unit_category
            )
        else:
            raise AttributeError(
                f"The transport technology {element.name} has neither "
                "opex_specific_fixed_per_distance nor opex_specific_fixed attribute."
            )
        return ParameterInputData(value, element.units.get(cls.name))
