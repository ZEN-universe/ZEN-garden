import numpy as np

from zen_garden.elements.storage_technology.expressions.existing_capacity_is_below_limit_energy import (  # noqa: E501
    ExistingCapacityIsBelowLimitEnergy,
)
from zen_garden.elements.storage_technology.masks import EnergyCapacityAvailableMask
from zen_garden.elements.storage_technology.parameters.capacity_limit_energy import (
    CapacityLimitEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_energy import (
    CapacityEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyCapacityLimitNotReachedConstraint(GenericConstraint):
    """Limit storage energy capacity where existing capacity leaves headroom."""

    name = "constraint_storage_energy_capacity_limit_not_reached"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the storage energy-capacity upper limit.

        Summary:
            Keep installed storage energy capacity below its finite limit.

        Formulation:

        .. math::

            E_{h,n,y}\leq\bar e_{h,n,y}
            \quad\text{if }e^{existing}_{h,n,y}<\bar e_{h,n,y}<\infty

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy-capacity upper-limit constraint.
        """
        model = model_constructor.optimization_model
        limit = model.parameters[CapacityLimitEnergy]
        selected = (limit != np.inf) & model.expressions[
            ExistingCapacityIsBelowLimitEnergy
        ]
        return model.variables[CapacityEnergy].where(selected) <= limit.where(
            selected, 0.0
        )
