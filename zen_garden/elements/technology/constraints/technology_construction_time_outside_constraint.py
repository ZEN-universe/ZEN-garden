"""Beyond-horizon investment constraint."""

from zen_garden.elements.technology.expressions.investments_beyond_horizon import (
    InvestmentsBeyondHorizon,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyConstructionTimeOutsideConstraint(GenericConstraint):
    """Prevent investments that commission only after the modeled horizon."""

    name = "constraint_technology_construction_time_outside"
    indices = (
        "set_technologies",
        "set_location",
        "set_time_steps_construction",
    )

    @classmethod
    def should_construct(cls, model_constructor):
        """Return whether any modeled investment lies beyond the horizon."""
        expression = model_constructor.optimization_model.expressions[
            InvestmentsBeyondHorizon
        ]
        return bool((expression.labels >= 0).any())

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct prevent investments that commission only after the modeled horizon.

        Summary:
            Prevent investments that commission only after the modeled horizon.

        Formulation:

        .. math::

            K^{investment}_{h,q,p,v}=0\\quad\\forall v+\\tau_h\\notin\\mathcal Y

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        return (
            model_constructor.optimization_model.expressions[InvestmentsBeyondHorizon]
            == 0
        )
