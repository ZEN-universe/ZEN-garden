from zen_garden.topology.generic_set import GenericSet


class SetInputCarriers(GenericSet):
    name, doc, index_set = (
        "set_input_carriers",
        "Input carriers indexed by conversion technology",
        "set_conversion_technologies",
    )

    @classmethod
    def get_data(cls, constructor):
        return constructor.element_registry.get_attribute_of_all_elements(
            constructor.element_class, "input_carrier"
        )
