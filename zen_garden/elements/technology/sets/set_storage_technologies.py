from zen_garden.topology.generic_set import GenericSet


class SetStorageTechnologies(GenericSet):
    name, doc = "set_storage_technologies", "Set of storage technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.model_schema.set_storage_technologies
