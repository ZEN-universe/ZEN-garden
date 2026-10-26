from zen_garden.model.component_types.set import GenericSet


class SetCapacityTypes(GenericSet):
    """Available technology capacity categories."""

    name = "set_capacity_types"
    symbol = r"c \in \mathcal{C}"
    doc = "Set of technology capacity types."

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.config.system.set_capacity_types
