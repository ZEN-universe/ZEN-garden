from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class DepreciationTime(GenericParameter):
    """Depreciation time of a newly built technology."""

    name = "depreciation_time"
    symbol = r"L^{\mathrm{dep}}_{h}"
    indices = ("set_technologies",)
    doc = (
        "Parameter which specifies the depreciation time of invested "
        "technologies (optional, default value is the lifetime)."
    )
    unit_category = {}
    dependencies = (Lifetime,)

    @classmethod
    def get_input_data(cls, element):
        """Load depreciation time or default it to technology lifetime."""
        if cls.name in element.element_data_loader.attribute_dict:
            value = element.element_data_loader.extract_input_data(
                cls.name, index_sets=[], unit_category={}
            )
            value[0] = max(element.config.system.interval_between_years, value[0])
        else:
            value = element.lifetime.copy()
        return ParameterInputData(value, element.units.get(cls.name))
