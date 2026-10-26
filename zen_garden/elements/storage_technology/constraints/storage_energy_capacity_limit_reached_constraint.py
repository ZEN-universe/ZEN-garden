import numpy as np

from zen_garden.elements.storage_technology.expressions.existing_capacity_is_below_limit_energy import (  # noqa: E501
    ExistingCapacityIsBelowLimitEnergy,
)
from zen_garden.elements.storage_technology.masks import (
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.elements.storage_technology.parameters.capacity_limit_energy import (
    CapacityLimitEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyCapacityLimitReachedConstraint(GenericConstraint):
    """Prevent energy additions where existing capacity reaches the limit."""

    name = "constraint_storage_energy_capacity_limit_reached"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAdditionEligibleMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the reached-limit storage energy restriction.

        Summary:
            Prevent storage energy additions where no capacity headroom remains.

        Formulation:

        .. math::

            \Delta E_{h,n,y}=0
            \quad\text{if }e^{existing}_{h,n,y}\geq\bar e_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint preventing unavailable storage energy additions.
        """
        model = model_constructor.optimization_model
        additions = model.variables[CapacityAdditionEnergy]
        if not model_constructor.config.system.allow_investment:
            return additions == 0
        limit = model.parameters[CapacityLimitEnergy]
        selected = (limit != np.inf) & ~model.expressions[
            ExistingCapacityIsBelowLimitEnergy
        ]
        return additions.where(selected) == 0
