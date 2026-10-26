from zen_garden.elements.storage_technology.masks import EnergyCapacityAvailableMask
from zen_garden.elements.storage_technology.parameters.capacity_lower_limit_energy import (  # noqa: E501
    CapacityLowerLimitEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_energy import (
    CapacityEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyCapacityLowerLimitConstraint(GenericConstraint):
    """Keep storage energy capacity above its configured minimum."""

    name = "constraint_storage_energy_capacity_lower_limit"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the minimum storage energy-capacity constraint.

        Summary:
            Require storage energy capacity to meet each positive lower limit.

        Formulation:

        .. math::

            E_{h,n,y}\geq\underline e_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Minimum storage energy-capacity constraint.
        """
        model = model_constructor.optimization_model
        lower_limit = model.parameters[CapacityLowerLimitEnergy]
        mask = lower_limit > 0.0
        return model.variables[CapacityEnergy].where(mask) >= lower_limit.where(
            mask, 0.0
        )
