from zen_garden.topology.generic_set import GenericSet


class SetInputCarriers(GenericSet):
    name, doc, index_set = (
        "set_input_carriers",
        "Input carriers indexed by conversion technology",
        "set_conversion_technologies",
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, "input_carrier"
        )
