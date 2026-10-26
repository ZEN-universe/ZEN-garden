import linopy as lp
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.storage_technology.expressions.surviving_capacity_additions_energy import (  # noqa: E501
    SurvivingCapacityAdditionsEnergy,
)
from zen_garden.elements.storage_technology.masks import EnergyCapacityAvailableMask
from zen_garden.elements.storage_technology.parameters.existing_capacities_energy import (  # noqa: E501
    ExistingCapacitiesEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_energy import (
    CapacityEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyLifetimeConstraint(GenericConstraint):
    """Define storage energy capacity from existing and surviving vintages."""

    name = "constraint_storage_energy_lifetime"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct end-of-year storage energy capacity.

        Summary:
            Combine existing and surviving modeled storage energy capacity.

        Formulation:

        .. math::

            E_{h,n,y}=e^{existing}_{h,n,y}
            +\sum_{v\in\mathcal V^{active}_{h,y}}\Delta E_{h,n,v}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            End-of-year storage energy-capacity constraint.
        """
        model = model_constructor.optimization_model
        lhs = lp.merge(
            [
                1 * model.variables[CapacityEnergy],
                -model.expressions[SurvivingCapacityAdditionsEnergy],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const, model.parameters[ExistingCapacitiesEnergy], join="left"
        )[1]
        return lhs == rhs
