from zen_garden.topology.generic_set import GenericSet


class SetRetrofittingBaseTechnologies(GenericSet):
    name, doc, index_set = (
        "set_retrofitting_base_technologies",
        "Base technologies indexed by retrofitting technology",
        "set_retrofitting_technologies",
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, "retrofit_base_technology"
        )
