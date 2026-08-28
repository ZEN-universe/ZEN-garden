from zen_garden.elements.technology.parameters._existing_quantity import (
    compute_existing_quantity,
)
from zen_garden.model.component_types.parameter import GenericParameter


class ExistingCapacities(GenericParameter):
    """Existing capacity still available at each investment year.

    Derived at construction time by aggregating :data:`capacity_existing` over
    the vintages whose remaining lifetime still covers the investment year. It
    is not loaded from input data and is rebuilt every rolling-horizon step
    because the underlying existing capacities change between steps.
    """

    name = "existing_capacities"
    indices = ("set_technologies", "set_capacity_types", "set_location", "set_years")
    doc = "Total available existing capacity at the optimization start"
    unit_category = {"energy_quantity": 1, "time": -1}
    dependencies = ["capacity_existing", "lifetime_existing", "lifetime"]

    @classmethod
    def store_input_data(cls, element):
        """No input data: derived from other parameters in :meth:`build`."""

    @classmethod
    def build(cls, model_constructor):
        model_constructor.optimization_model.add_parameter(
            name=cls.name,
            doc=cls.doc,
            data=compute_existing_quantity(model_constructor, "capacity_existing"),
        )
