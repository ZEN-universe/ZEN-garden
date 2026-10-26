from zen_garden.elements.storage_technology.masks import (
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.elements.storage_technology.parameters.capacity_addition_min_energy import (  # noqa: E501
    CapacityAdditionMinEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.storage_technology.variables.technology_installation_energy import (  # noqa: E501
    TechnologyInstallationEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyMinCapacityAdditionConstraint(GenericConstraint):
    """Enforce the minimum storage energy-capacity addition."""

    name = "constraint_storage_energy_min_capacity_addition"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAdditionEligibleMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        minimum = model_constructor.optimization_model.parameters[
            CapacityAdditionMinEnergy
        ]
        return bool(((minimum != 0) & minimum.notnull()).any().item())

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the minimum storage energy addition.

        Summary:
            Apply the minimum energy addition when an installation occurs.

        Formulation:

        .. math::

            \underline{\Delta e}_{h}z^{install,E}_{h,n,y}
            \leq\Delta E_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Minimum storage energy-capacity addition constraint.
        """
        model = model_constructor.optimization_model
        minimum = model.parameters[CapacityAdditionMinEnergy]
        mask = (minimum != 0) & minimum.notnull()
        lhs = mask * (
            minimum * model.variables[TechnologyInstallationEnergy]
            - model.variables[CapacityAdditionEnergy]
        )
        return lhs <= 0
