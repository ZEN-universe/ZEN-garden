from zen_garden.elements.technology.parameters._existing_quantity import (
    compute_existing_quantity,
)
from zen_garden.elements.technology.parameters.capacity_existing import CapacityExisting
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.model.component_types.parameter import GenericParameter, ParameterData


class ExistingCapacities(GenericParameter):
    """Existing capacity still available at each investment year.

    Derived at construction time by aggregating :data:`capacity_existing` over
    the vintages whose remaining lifetime still covers the investment year. It
    is not loaded from input data and is rebuilt every rolling-horizon step
    because the underlying existing capacities change between steps.
    """

    name = "existing_capacities"
    symbol = r"k^{\mathrm{existing}}_{h,q,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = (
        "Installed capacity that remains from investments made before the "
        "modeled horizon."
    )
    unit_category = {"energy_quantity": 1, "time": -1}
    dependencies = (CapacityExisting, LifetimeExisting, Lifetime)

    @classmethod
    def get_input_data(cls, element):
        """Return no input because the parameter is derived."""
        return None

    @classmethod
    def get_parameter_value(cls, model_constructor):
        """Return existing capacity available in each modeled year."""
        return ParameterData(
            compute_existing_quantity(model_constructor, "capacity_existing"),
            cls.indices,
        )
