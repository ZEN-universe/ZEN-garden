import numpy as np

from zen_garden.elements.energy_system.parameters.knowledge_spillover_rate import (
    KnowledgeSpilloverRate,
)
from zen_garden.elements.storage_technology.expressions.capacity_growth_from_existing_capacity_with_spillovers_energy import (  # noqa: E501
    CapacityGrowthFromExistingCapacityWithSpilloversEnergy,
)
from zen_garden.elements.storage_technology.expressions.capacity_growth_from_spillovers_energy import (  # noqa: E501
    CapacityGrowthFromSpilloversEnergy,
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


class StorageEnergyDiffusionLimitLocationConstraint(GenericConstraint):
    """Limit storage energy additions at each node using market knowledge."""

    name = "constraint_storage_energy_diffusion_limit"
    doc = r"""Limit nodal storage-energy additions with fully expanded learning terms.

    .. math::

        \begin{aligned}
        \Delta E_{h,n,y}\leq{}&d_{h,n,y}\sum_i\left[
        e^{existing}_{h,n,i}+s_h\sum_{n'\ne n}e^{existing}_{h,n',i}\right]
        (1-\delta_h)^{\Delta y(y-y_0)-1+\tau_h-\tau^{existing}_{h,n,i}}\\
        &+d_{h,n,y}\sum_{v<y}\left[\Delta E_{h,n,v}
        +s_h\sum_{n'\ne n}\Delta E_{h,n',v}\right]
        (1-\delta_h)^{\Delta y(y-1-v)}\\
        &+\sum_{h'\ne h:c_{h'}^{ref}=c_h^{ref}}s^{market}
        (E_{h',n,y}-\Delta E_{h',n,y})+e^{unbounded}_h.
        \end{aligned}
    """
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        model = model_constructor.optimization_model
        return model.parameters[KnowledgeSpilloverRate] != np.inf and bool(
            (storage_energy_diffusion_rate(model_constructor) != np.inf).any().item()
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the nodal storage energy diffusion limit.

        Summary:
            Limit energy additions at each node using local and shared knowledge.

        Formulation:

        .. math::

            \Delta E_{h,n,y}\leq d_{h,n,y}e^{knowledge,spill}_{h,n,y}
            +E^{peer}_{h,n,y}+e^{unbounded}_{h}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Nodal storage energy diffusion constraint.
        """
        model = model_constructor.optimization_model
        lhs = (
            model.variables[CapacityAdditionEnergy]
            - model.expressions[CapacityGrowthFromSpilloversEnergy]
            - model.expressions[CapacityGrowthFromUnboundedMarketEnergy]
        )
        free_growth = model.parameters[CapacityAdditionUnboundedEnergy].broadcast_like(
            storage_energy_diffusion_rate(model_constructor)
        )
        rhs = (
            model.expressions[CapacityGrowthFromExistingCapacityWithSpilloversEnergy]
            + free_growth
        ).broadcast_like(lhs.const)
        mask = storage_energy_diffusion_rate(model_constructor) != np.inf
        return align_and_mask(lhs, mask) <= align_and_mask(rhs, mask)
