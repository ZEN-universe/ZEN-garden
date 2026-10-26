"""Storage energy-capacity headroom indicator."""

from zen_garden.elements.storage_technology.parameters.capacity_limit_energy import (
    CapacityLimitEnergy,
)
from zen_garden.elements.storage_technology.parameters.existing_capacities_energy import (  # noqa: E501
    ExistingCapacitiesEnergy,
)
from zen_garden.model.component_types.expression import GenericExpression


class ExistingCapacityIsBelowLimitEnergy(GenericExpression):
    """Indicate where existing storage energy capacity is below its limit."""

    name = "existing_capacity_is_below_limit_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct storage energy-capacity headroom.

        Summary:
            Compare existing storage energy capacity with its upper limit.

        Formulation:

        .. math::

            e^{existing}_{h,n,y}<\bar e_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Boolean storage energy-capacity headroom indicator.
        """
        parameters = model_constructor.optimization_model.parameters
        return parameters[ExistingCapacitiesEnergy] < parameters[CapacityLimitEnergy]
