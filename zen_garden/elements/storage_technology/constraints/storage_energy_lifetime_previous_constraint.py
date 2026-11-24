from zen_garden.elements.storage_technology.expressions.capacity_previous_energy import (  # noqa: E501
    CapacityPreviousEnergy,
)
from zen_garden.elements.storage_technology.masks import EnergyCapacityAvailableMask
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyLifetimePreviousConstraint(GenericConstraint):
    """Define storage energy capacity before current-year additions."""

    name = "constraint_storage_energy_lifetime_previous"
    doc = r"""Keep beginning-of-year storage energy capacity non-negative.

    .. math::

        E_{h,n,y}-\Delta E_{h,n,y}\geq 0
    """
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct beginning-of-year storage energy capacity.

        Summary:
            Exclude current-year additions from available storage energy capacity.

        Formulation:

        .. math::

            E^{previous}_{h,n,y}=e^{existing}_{h,n,y}
            +\sum_{v\in\mathcal V^{active}_{h,y}}\Delta E_{h,n,v}
            -\Delta E_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Beginning-of-year storage energy-capacity constraint.
        """
        return (
            model_constructor.optimization_model.expressions[CapacityPreviousEnergy]
            >= 0
        )
