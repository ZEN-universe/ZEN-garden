import numpy as np

from zen_garden.elements.technology.masks import CapacityAdditionEligibleMask
from zen_garden.elements.technology.parameters.capacity_addition_max import (
    CapacityAdditionMax,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.elements.technology.variables.technology_installation import (
    TechnologyInstallation,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyMaxCapacityAdditionConstraint(GenericConstraint):
    """Technology Max Capacity Addition."""

    name = "constraint_technology_max_capacity_addition"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAdditionEligibleMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        maximum = model_constructor.optimization_model.parameters[CapacityAdditionMax]
        mask = (maximum != np.inf) & (maximum != 0) & maximum.notnull()
        return bool(mask.any().item())

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Max capacity addition of technology.

        Formulation:

        .. math::
            \\overline{\\Delta k}_{h} g_{h,p,y}
            \\geq \\Delta K_{h,p,y}

        The constraint is omitted for zero, missing, or infinite maximum additions.

        Notation:

        :math:`\\overline{\\Delta k}_{h}`: maximum capacity addition for technology
        :math:`h` (with capacity type implicit)
        :math:`g_{h,p,y}`: binary variable which equals 1 if technology is installed
        at location :math:`p` in year :math:`y`
        :math:`\\Delta K_{h,p,y}`: capacity addition

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        capacity_addition_max = model_constructor.optimization_model.parameters[
            CapacityAdditionMax
        ]
        mask = (
            (capacity_addition_max != np.inf)
            & (capacity_addition_max != 0)
            & (capacity_addition_max.notnull())
        )

        lhs = mask * (
            capacity_addition_max
            * model_constructor.optimization_model.variables[TechnologyInstallation]
            - model_constructor.optimization_model.variables[CapacityAddition]
        )
        rhs = 0
        constraints = lhs >= rhs

        return constraints
