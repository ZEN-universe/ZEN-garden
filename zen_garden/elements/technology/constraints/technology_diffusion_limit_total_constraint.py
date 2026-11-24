import numpy as np

from zen_garden.elements.technology.diffusion import (
    diffusion_rate,
    technology_location_mask,
)
from zen_garden.elements.technology.expressions.capacity_growth_from_existing_capacity import (  # noqa: E501
    CapacityGrowthFromExistingCapacity,
)
from zen_garden.elements.technology.expressions.capacity_growth_from_local_learning import (  # noqa: E501
    CapacityGrowthFromLocalLearning,
)
from zen_garden.elements.technology.expressions.capacity_growth_from_unbounded_market import (  # noqa: E501
    CapacityGrowthFromUnboundedMarket,
)
from zen_garden.elements.technology.parameters.capacity_addition_unbounded import (
    CapacityAdditionUnbounded,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.linear_expression import align_and_mask


class TechnologyDiffusionLimitTotalConstraint(GenericConstraint):
    """Limit aggregate additions without counting cross-location spillovers twice."""

    name = "constraint_technology_diffusion_limit_total"
    indices = ("set_technologies", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return bool((diffusion_rate(model_constructor) != np.inf).any().item())

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct the model component.

        Summary:
            Limit aggregate additions without counting cross-location spillovers twice.

        Formulation:

        .. math::

            \\begin{aligned}
            \\sum_p \\Delta K_{h,q,p,y} \\leq \\sum_p \\Bigg[{}
            &d_{h,p,y} \\sum_{h'} k^{\\mathrm{existing}}_{h',p}
            (1-\\delta_h)^{\\Delta y(y-y_0)-1+\\tau_h-\\tau^{\\mathrm{existing}}_{h'}}
            \\\\
            &+d_{h,p,y} \\sum_{v<y} \\Delta K_{h,q,p,v}
            (1-\\delta_h)^{\\Delta y(y-1-v)} \\\\
            &+\\sum_{h' \\ne h:c_{h'}^{\\mathrm{ref}}=c_h^{\\mathrm{ref}}}
            s^{\\mathrm{market}}(K_{h',q,p,y}-\\Delta K_{h',q,p,y})
            +k^{\\mathrm{unbounded}}_{h,q,p,y}\\Bigg].
            \\end{aligned}

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
        additions = model.variables[CapacityAddition]
        lhs = (
            additions
            - model.expressions[CapacityGrowthFromLocalLearning]
            - model.expressions[CapacityGrowthFromUnboundedMarket]
        ).sum("set_location")
        free_growth = (
            model.parameters[CapacityAdditionUnbounded]
            .broadcast_like(diffusion_rate(model_constructor))
            .where(technology_location_mask(model_constructor), 0)
        )
        rhs = (model.expressions[CapacityGrowthFromExistingCapacity] + free_growth).sum(
            "set_location"
        )
        rhs = rhs.broadcast_like(lhs.const)
        mask = diffusion_rate(model_constructor).sum("set_location") != np.inf
        return align_and_mask(lhs, mask) <= align_and_mask(rhs, mask)
