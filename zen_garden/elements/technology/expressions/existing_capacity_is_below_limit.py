"""Capacity-limit headroom indicator."""

from zen_garden.elements.technology.parameters.capacity_limit import CapacityLimit
from zen_garden.elements.technology.parameters.existing_capacities import (
    ExistingCapacities,
)
from zen_garden.model.component_types.expression import GenericExpression


class ExistingCapacityIsBelowLimit(GenericExpression):
    """Indicate where existing capacity is below the configured upper limit."""

    name = "existing_capacity_is_below_limit"
    unit_category = {}
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Indicate where existing capacity is below the configured upper limit.

        Formulation:

        .. math::

            k^{existing}_{h,q,p,y}<\\bar k_{h,q,p,y}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        parameters = model_constructor.optimization_model.parameters
        return parameters[ExistingCapacities] < parameters[CapacityLimit]
