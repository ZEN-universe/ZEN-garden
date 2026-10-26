from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class LifetimeExisting(GenericParameter):
    """Parameter specifying the remaining lifetime of an existing technology."""

    name = "lifetime_existing"
    symbol = r"L^{\mathrm{ex}}_{h,v}"
    indices = ("set_technologies", "set_location", "set_technologies_existing")
    doc = "Parameter which specifies the remaining lifetime of an existing technology."
    unit_category = {}
    dependencies = (Lifetime,)

    @classmethod
    def get_input_data(cls, element):
        """Extract remaining lifetime for existing capacity vintages."""
        value = element.element_data_loader.extract_lifetime_existing(
            "capacity_existing", index_sets=cls._input_indices(element)
        )
        return ParameterInputData(value, element.units.get(cls.name))
