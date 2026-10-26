import numpy as np

from zen_garden.elements.storage_technology.masks import (
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.elements.storage_technology.parameters.capacity_addition_max_energy import (  # noqa: E501
    CapacityAdditionMaxEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.storage_technology.variables.technology_installation_energy import (  # noqa: E501
    TechnologyInstallationEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyMaxCapacityAdditionConstraint(GenericConstraint):
    """Enforce the maximum storage energy-capacity addition."""

    name = "constraint_storage_energy_max_capacity_addition"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAdditionEligibleMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        maximum = model_constructor.optimization_model.parameters[
            CapacityAdditionMaxEnergy
        ]
        mask = (maximum != np.inf) & (maximum != 0) & maximum.notnull()
        return bool(mask.any().item())

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the maximum storage energy addition.

        Summary:
            Apply the maximum energy addition when an installation occurs.

        Formulation:

        .. math::

            \overline{\Delta e}_{h}z^{install,E}_{h,n,y}
            \geq\Delta E_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Maximum storage energy-capacity addition constraint.
        """
        model = model_constructor.optimization_model
        maximum = model.parameters[CapacityAdditionMaxEnergy]
        mask = (maximum != np.inf) & (maximum != 0) & maximum.notnull()
        lhs = mask * (
            maximum * model.variables[TechnologyInstallationEnergy]
            - model.variables[CapacityAdditionEnergy]
        )
        return lhs >= 0
