from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.set import GenericDomainSet


class SetRetrofittingTechnologies(GenericDomainSet):
    name = "set_retrofitting_technologies"
    symbol = r"h \in \mathcal{H}^{\mathrm{retro}}"
    doc = "Set of retrofitting technologies."
    subset_of = SetConversionTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_retrofitting_technologies
