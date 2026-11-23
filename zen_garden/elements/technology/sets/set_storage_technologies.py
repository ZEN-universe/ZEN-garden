from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.model.component_types.set import GenericDomainSet


class SetStorageTechnologies(GenericDomainSet):
    name = "set_storage_technologies"
    symbol = r"h \in \mathcal{H}^{\mathrm{stor}}"
    doc = "Set of storage technologies."
    subset_of = SetTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_storage_technologies
