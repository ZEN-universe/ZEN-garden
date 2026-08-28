from zen_garden.topology.generic_set import GenericSet


class SetTechnologiesExisting(GenericSet):
    name, doc, index_set = (
        "set_technologies_existing",
        "Set of existing technology vintages",
        "set_technologies",
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, cls.name
        )
