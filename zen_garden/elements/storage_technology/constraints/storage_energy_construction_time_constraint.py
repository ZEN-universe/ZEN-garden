import linopy as lp
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.storage_technology.expressions.capacity_additions_from_new_investments_energy import (  # noqa: E501
    CapacityAdditionsFromNewInvestmentsEnergy,
)
from zen_garden.elements.storage_technology.expressions.capacity_additions_from_pre_horizon_investments_energy import (  # noqa: E501
    CapacityAdditionsFromPreHorizonInvestmentsEnergy,
)
from zen_garden.elements.storage_technology.masks import (
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyConstructionTimeConstraint(GenericConstraint):
    """Link storage energy additions to investments using construction time."""

    name = "constraint_storage_energy_construction_time"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAdditionEligibleMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct storage energy-capacity commissioning.

        Summary:
            Link storage energy additions to earlier investment decisions.

        Formulation:

        .. math::

            \Delta E_{h,n,y}=
            \sum_{v:v+\tau_h=y}E^{investment}_{h,n,v}
            +\sum_{v\notin\mathcal Y:v+\tau_h=y}
            e^{investment,existing}_{h,n,v}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy-capacity commissioning constraint.
        """
        model = model_constructor.optimization_model
        lhs = lp.merge(
            [
                1 * model.variables[CapacityAdditionEnergy],
                -model.expressions[CapacityAdditionsFromNewInvestmentsEnergy],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const,
            model.expressions[CapacityAdditionsFromPreHorizonInvestmentsEnergy],
            join="left",
        )[1]
        return lhs == rhs
