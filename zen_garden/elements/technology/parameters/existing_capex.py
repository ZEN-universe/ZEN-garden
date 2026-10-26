from zen_garden.elements.technology.parameters._existing_quantity import (
    compute_existing_quantity,
)
from zen_garden.elements.technology.parameters.capex_capacity_existing import (
    CapexCapacityExisting,
)
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.model.component_types.parameter import GenericParameter, ParameterData


class ExistingCapex(GenericParameter):
    """Outstanding capex of existing capacity still available at each year.

    Derived at construction time by aggregating :data:`capex_capacity_existing`
    over the vintages whose remaining lifetime still covers the investment year.
    It is not loaded from input data and is rebuilt every rolling-horizon step
    because the underlying existing capacities change between steps.
    """

    name = "existing_capex"
    symbol = r"k^{\mathrm{existing,capex}}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Overnight capital cost associated with existing capacity."
    unit_category = {"money": 1}
    dependencies = (CapexCapacityExisting, LifetimeExisting, Lifetime)

    @classmethod
    def get_input_data(cls, element):
        """Return no input because the parameter is derived."""
        return None

    @classmethod
    def get_parameter_value(cls, model_constructor):
        """Return outstanding CAPEX available in each modeled year."""
        return ParameterData(
            compute_existing_quantity(model_constructor, "capex_capacity_existing"),
            cls.indices,
        )
