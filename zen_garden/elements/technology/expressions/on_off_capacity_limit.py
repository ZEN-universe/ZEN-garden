"""Finite Big-M capacity limit for technology on/off constraints."""

import numpy as np

from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.elements.technology.variables.tech_on_var import TechOnVar
from zen_garden.model.component_types.expression import GenericExpression


class OnOffCapacityLimit(GenericExpression):
    """Return and validate capacity upper bounds used by on/off linearization."""

    name = "on_off_capacity_limit"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        technologies = model_constructor.optimization_model.sets.on_off_technologies()
        return len(technologies) > 0

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Return and validate capacity upper bounds used by on/off linearization.

        Formulation:

        .. math::

            \\bar K_{h,power,p,y}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        capacity_limit = optimization_model.variables[Capacity].upper
        binary_mask = optimization_model.variables[TechOnVar].mask
        infinite = (capacity_limit.where(binary_mask) == np.inf).to_series()
        invalid = infinite[infinite].index.droplevel(2).unique().to_list()
        if infinite.any():
            raise ValueError(
                f"On/off capacity limit is infinite for {invalid}. Configure finite "
                "capacity limits for these technologies."
            )
        return capacity_limit
