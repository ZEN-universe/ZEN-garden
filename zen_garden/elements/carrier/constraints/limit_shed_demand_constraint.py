import numpy as np

from zen_garden.elements.carrier.parameters.demand import Demand
from zen_garden.elements.carrier.parameters.price_shed_demand import PriceShedDemand
from zen_garden.elements.carrier.variables.shed_demand import ShedDemand
from zen_garden.model.component_types.constraint import GenericConstraint


class LimitShedDemandConstraint(GenericConstraint):
    """Limit shedding to demand, or to zero when shedding is disabled."""

    name = "constraint_limit_shed_demand"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct limit shedding to demand, or to zero when shedding is disabled.

        Summary:
            Limit shedding to demand, or to zero when shedding is disabled.

        Formulation:

        .. math::

            F^{shed}_{c,n,y,r}\\leq d_{c,n,y,r}

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
        allowed = model.parameters[PriceShedDemand] != np.inf
        return model.variables[ShedDemand] <= model.parameters[Demand].where(
            allowed, 0.0
        )
