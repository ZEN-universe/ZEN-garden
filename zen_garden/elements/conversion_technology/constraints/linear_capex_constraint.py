import numpy as np

from zen_garden.elements.conversion_technology.expressions.conversion_overnight_capex import (  # noqa: E501
    ConversionOvernightCapex,
)
from zen_garden.elements.conversion_technology.parameters.capex_specific_conversion import (  # noqa: E501
    CapexSpecificConversion,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.masks import PowerCapacityAdditionMask
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.variables.cost_capex_overnight import (
    CostCapexOvernight,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class LinearCapexConstraint(GenericConstraint):
    """Set conversion overnight CAPEX from power-capacity additions."""

    name = "constraint_linear_capex"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (PowerCapacityAdditionMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct set conversion overnight capex from power-capacity additions.

        Summary:
            Set conversion overnight CAPEX from power-capacity additions.

        Formulation:

        .. math::

            C^{capex,overnight}=k^{capex,conv}\\Delta K

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
        technologies = model.sets[SetConversionTechnologies]
        nodes = model.sets[SetNodes]
        cost = model.variables[CostCapexOvernight].sel(
            set_technologies=technologies,
            set_location=nodes,
        )
        specific = (
            model.parameters[CapexSpecificConversion]
            .rename(
                {
                    old: new
                    for old, new in zip(
                        model.parameters[CapexSpecificConversion].dims,
                        ["set_technologies", "set_location", "set_years"],
                        strict=False,
                    )
                }
            )
            .broadcast_like(cost.lower)
        )
        mask = ~np.isnan(specific)
        return cost.where(mask) == model.expressions[ConversionOvernightCapex]
