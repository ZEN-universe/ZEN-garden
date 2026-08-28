from zen_garden.model.component_types.set import GenericSet


class SetReferenceCarriers(GenericSet):
    name, doc, index_set = (
        "set_reference_carriers",
        "Reference carriers indexed by technology",
        "set_technologies",
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, "reference_carrier"
        )
