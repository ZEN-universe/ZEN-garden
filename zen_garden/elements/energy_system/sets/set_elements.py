from zen_garden.model.component_types.set import GenericSet


class SetElements(GenericSet):
    name, doc, indexing_set = "set_elements", "Set of elements", True

    @classmethod
    def get_data(cls, model_constructor):
        return list(
            set(model_constructor.model_schema.set_technologies)
            | set(model_constructor.model_schema.set_carriers)
        )
