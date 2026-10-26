from zen_garden.model.component_types.set import GenericSet


class SetStorageTechnologies(GenericSet):
    name = "set_storage_technologies"
    symbol = r"h \in \mathcal{H}^{\mathrm{stor}}"
    doc = "Set of storage technologies."

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_storage_technologies
