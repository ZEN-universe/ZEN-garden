from zen_garden.elements.technology.masks import (
    OnOffTechnologyMask,
    PowerOperationMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class TechOnVar(GenericVariable):
    """Variable for technology on/off binary."""

    name = "tech_on_var"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = (
        "Binary variable indicating when technology is switched on at location l and "
        "time t"
    )
    unit_category = {}
    binary = True
    masks = [PowerOperationMask, OnOffTechnologyMask]

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
