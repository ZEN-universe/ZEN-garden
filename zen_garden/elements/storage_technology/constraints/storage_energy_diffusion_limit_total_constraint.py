import numpy as np

from zen_garden.elements.storage_technology.expressions.capacity_growth_from_existing_capacity_energy import (  # noqa: E501
    CapacityGrowthFromExistingCapacityEnergy,
)
from zen_garden.elements.storage_technology.expressions.capacity_growth_from_local_learning_energy import (  # noqa: E501
    CapacityGrowthFromLocalLearningEnergy,
)
from zen_garden.elements.storage_technology.expressions.capacity_growth_from_unbounded_market_energy import (  # noqa: E501
    CapacityGrowthFromUnboundedMarketEnergy,
)
from zen_garden.elements.storage_technology.parameters.capacity_addition_unbounded_energy import (  # noqa: E501
    CapacityAdditionUnboundedEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.diffusion import storage_energy_diffusion_rate
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.linear_expression import align_and_mask


class StorageEnergyDiffusionLimitTotalConstraint(GenericConstraint):
    """Limit aggregate storage energy additions without double-counting spillovers."""

    name = "constraint_storage_energy_diffusion_limit_total"
    indices = ("set_storage_technologies", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(
            (storage_energy_diffusion_rate(model_constructor) != np.inf).any().item()
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the aggregate storage energy diffusion limit.

        Summary:
            Limit total energy additions using existing and learned market capacity.

        Formulation:

        .. math::

            \sum_n\Delta E_{h,n,y}\leq\sum_n\left[
            d_{h,n,y}e^{knowledge}_{h,n,y}
            +E^{peer}_{h,n,y}+e^{unbounded}_{h}\right]

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Aggregate storage energy diffusion constraint.
        """
        model = model_constructor.optimization_model
        lhs = (
            model.variables[CapacityAdditionEnergy]
            - model.expressions[CapacityGrowthFromLocalLearningEnergy]
            - model.expressions[CapacityGrowthFromUnboundedMarketEnergy]
        ).sum("set_nodes")
        free_growth = model.parameters[CapacityAdditionUnboundedEnergy].broadcast_like(
            storage_energy_diffusion_rate(model_constructor)
        )
        rhs = (
            model.expressions[CapacityGrowthFromExistingCapacityEnergy] + free_growth
        ).sum("set_nodes")
        rhs = rhs.broadcast_like(lhs.const)
        mask = (
            storage_energy_diffusion_rate(model_constructor).sum("set_nodes") != np.inf
        )
        return align_and_mask(lhs, mask) <= align_and_mask(rhs, mask)
