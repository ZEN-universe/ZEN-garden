from zen_garden.elements.storage_technology.expressions.investments_beyond_horizon_energy import (  # noqa: E501
    InvestmentsBeyondHorizonEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageEnergyConstructionTimeOutsideConstraint(GenericConstraint):
    """Prevent storage energy investments completed after the horizon."""

    name = "constraint_storage_energy_construction_time_outside"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_time_steps_construction",
    )

    @classmethod
    def should_construct(cls, model_constructor):
        expression = model_constructor.optimization_model.expressions[
            InvestmentsBeyondHorizonEnergy
        ]
        return bool((expression.labels >= 0).any())

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct the beyond-horizon storage energy restriction.

        Summary:
            Prevent investments that cannot finish inside the modeled horizon.

        Formulation:

        .. math::

            E^{investment}_{h,n,v}=0
            \quad\forall v+\tau_h\notin\mathcal Y

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Beyond-horizon storage energy investment constraint.
        """
        return (
            model_constructor.optimization_model.expressions[
                InvestmentsBeyondHorizonEnergy
            ]
            == 0
        )
