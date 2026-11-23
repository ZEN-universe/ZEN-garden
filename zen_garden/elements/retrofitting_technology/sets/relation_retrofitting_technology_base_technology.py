from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.sets.set_retrofitting_technologies import (
    SetRetrofittingTechnologies,
)
from zen_garden.model.component_types.set import GenericRelationSet


class RelationRetrofittingTechnologyBaseTechnology(GenericRelationSet):
    """Base conversion technologies associated with retrofit technologies."""

    name = "relation_retrofitting_technology_base_technology"
    dimensions = (SetRetrofittingTechnologies, SetConversionTechnologies)
    apply_by_default = False

    @classmethod
    def get_combinations(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class,
            "retrofit_base_technology",
        )
