from zen_garden.model.component_types.parameter import GenericParameter


class CapexSpecificTransport(GenericParameter):
    """Capex per capacity unit."""

    name = "capex_specific_transport"
    indices = ("set_transport_technologies", "set_edges", "set_years")
    doc = "Capex per capacity unit"
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
    dependencies = ["distance"]

    @classmethod
    def store_input_data(cls, element):
        """Load either specific or distance-based transport capex."""
        attributes = element.element_data_loader.attribute_dict
        indices = ["set_edges", "set_years"]
        specific_units = {"money": 1, "energy_quantity": -1, "time": 1}
        distance_units = {"money": 1, "distance": -1}

        if element.config.system.double_capex_transport:
            specific = element.element_data_loader.extract_input_data(
                "capex_specific_transport", indices, specific_units
            )
            per_distance = element.element_data_loader.extract_input_data(
                "capex_per_distance_transport", indices, distance_units
            )
        elif "capex_per_distance_transport" in attributes:
            per_distance_input = element.element_data_loader.extract_input_data(
                "capex_per_distance_transport",
                indices,
                {
                    "money": 1,
                    "distance": -1,
                    "energy_quantity": -1,
                    "time": 1,
                },
            )
            specific = per_distance_input * element.distance
            per_distance = specific * 0.0
        elif "capex_specific_transport" in attributes:
            specific = element.element_data_loader.extract_input_data(
                "capex_specific_transport", indices, specific_units
            )
            per_distance = specific * 0.0
        else:
            raise AttributeError(
                f"The transport technology {element.name} has neither "
                "capex_per_distance_transport nor capex_specific_transport attribute."
            )

        cls._store_value(element, cls.name, specific)
        cls._store_value(element, "capex_per_distance_transport", per_distance)
