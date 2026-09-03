from zen_garden.model.component_types.set import GenericSet


class SetOutputCarriers(GenericSet):
    name, doc, index_set = (
        "set_output_carriers",
        "Output carriers indexed by conversion technology",
        "set_conversion_technologies",
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, "output_carrier"
        )
