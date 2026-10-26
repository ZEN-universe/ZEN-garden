import numpy as np

from zen_garden.elements.energy_system.parameters.knowledge_spillover_rate import (
    KnowledgeSpilloverRate,
)
from zen_garden.elements.technology.diffusion import (
    diffusion_rate,
    technology_location_mask,
)
from zen_garden.elements.technology.expressions.capacity_growth_from_existing_capacity_with_spillovers import (  # noqa: E501
    CapacityGrowthFromExistingCapacityWithSpillovers,
)
from zen_garden.elements.technology.expressions.capacity_growth_from_spillovers import (
    CapacityGrowthFromSpillovers,
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


class TechnologyDiffusionLimitLocationConstraint(GenericConstraint):
    """Limit additions at each location using available installation knowledge."""

    name = "constraint_technology_diffusion_limit"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        model = model_constructor.optimization_model
        return model.parameters[KnowledgeSpilloverRate] != np.inf and bool(
            (diffusion_rate(model_constructor) != np.inf).any().item()
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct the model component.

        Summary:
            Limit additions at each location using available installation knowledge.

        Formulation:

        .. math::

            \\begin{aligned}
            \\Delta K_{h,q,p,y} \\leq {}&
            d_{h,p,y} \\sum_{h'}
            \\left[k^{\\mathrm{existing}}_{h',p}
            +s_h \\sum_{p' \\ne p} k^{\\mathrm{existing}}_{h',p'}\\right]
            (1-\\delta_h)^{\\Delta y(y-y_0)-1+\\tau_h-\\tau^{\\mathrm{existing}}_{h'}}
            \\\\
            &+d_{h,p,y} \\sum_{v<y}
            \\left[\\Delta K_{h,q,p,v}
            +s_h \\sum_{p' \\ne p} \\Delta K_{h,q,p',v}\\right]
            (1-\\delta_h)^{\\Delta y(y-1-v)} \\\\
            &+\\sum_{h' \\ne h:c_{h'}^{\\mathrm{ref}}=c_h^{\\mathrm{ref}}}
            s^{\\mathrm{market}}K^{\\mathrm{previous}}_{h',q,p,y}
            +k^{\\mathrm{unbounded}}_{h,q,p,y}.
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
        lhs = (
            model.variables[CapacityAddition]
            - model.expressions[CapacityGrowthFromSpillovers]
            - model.expressions[CapacityGrowthFromUnboundedMarket]
        )
        free_growth = (
            model.parameters[CapacityAdditionUnbounded]
            .broadcast_like(diffusion_rate(model_constructor))
            .where(technology_location_mask(model_constructor), 0)
        )
        rhs = (
            model.expressions[CapacityGrowthFromExistingCapacityWithSpillovers]
            + free_growth
        ).broadcast_like(lhs.const)
        mask = diffusion_rate(model_constructor) != np.inf
        return align_and_mask(lhs, mask) <= align_and_mask(rhs, mask)
