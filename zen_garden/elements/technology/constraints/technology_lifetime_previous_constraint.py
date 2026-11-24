"""Beginning-of-year available-capacity constraint."""

from zen_garden.elements.technology.expressions.capacity_previous import (
    CapacityPrevious,
)
from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyLifetimePreviousConstraint(GenericConstraint):
    """Define capacity before additions made in the current year."""

    name = "constraint_technology_lifetime_previous"
    doc = r"""Keep beginning-of-year power capacity non-negative.

    .. math::

        K_{h,p,y}-\Delta K_{h,p,y}\geq 0
    """
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct define capacity before additions made in the current year.

        Summary:
            Define capacity before additions made in the current year.

        Formulation:

        .. math::

            K^{previous} =k^{existing} + \\sum_{v \\in \\mathcal V^{active}} \\Delta
            K_v- \\Delta K_y

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        return model_constructor.optimization_model.expressions[CapacityPrevious] >= 0
