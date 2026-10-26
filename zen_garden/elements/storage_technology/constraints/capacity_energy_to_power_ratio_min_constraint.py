import numpy as np

from zen_garden.elements.storage_technology.expressions.storage_energy_capacity_addition import (  # noqa: E501
    StorageEnergyCapacityAddition,
)
from zen_garden.elements.storage_technology.expressions.storage_power_capacity_addition import (  # noqa: E501
    StoragePowerCapacityAddition,
)
from zen_garden.elements.storage_technology.parameters.energy_to_power_ratio_min import (  # noqa: E501
    EnergyToPowerRatioMin,
)
from zen_garden.elements.technology.masks import CapacityAdditionAnyTypeMask
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CapacityEnergyToPowerRatioMinConstraint(GenericConstraint):
    """Enforce the minimum energy-to-power ratio of storage additions."""

    name = "constraint_capacity_energy_to_power_ratio_min"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (CapacityAdditionAnyTypeMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct enforce the minimum energy-to-power ratio of storage additions.

        Summary:
            Enforce the minimum energy-to-power ratio of storage additions.

        Formulation:

        .. math::

            \\Delta K_{energy}\\geq\\underline\\rho\\Delta K_{power}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        ratio = model.parameters[EnergyToPowerRatioMin]
        mask = ratio != np.inf
        return model.expressions[StorageEnergyCapacityAddition].where(mask) >= (
            ratio * model.expressions[StoragePowerCapacityAddition]
        ).where(mask)
