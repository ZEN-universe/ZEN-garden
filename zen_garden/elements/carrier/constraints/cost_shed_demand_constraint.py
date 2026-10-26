import numpy as np

from zen_garden.elements.carrier.parameters.price_shed_demand import PriceShedDemand
from zen_garden.elements.carrier.variables.cost_shed_demand import CostShedDemand
from zen_garden.elements.carrier.variables.shed_demand import ShedDemand
from zen_garden.model.component_types.constraint import GenericConstraint


class CostShedDemandConstraint(GenericConstraint):
    """Calculate the hourly cost of shedding demand."""

    name = "constraint_cost_shed_demand"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct calculate the hourly cost of shedding demand.

        Summary:
            Calculate the hourly cost of shedding demand.

        Formulation:

        .. math::

            C^{shed}=p^{shed}F^{shed}

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
        price = model.parameters[PriceShedDemand]
        return (
            model.variables[CostShedDemand] - price * model.variables[ShedDemand]
        ).where(price != np.inf) == 0
