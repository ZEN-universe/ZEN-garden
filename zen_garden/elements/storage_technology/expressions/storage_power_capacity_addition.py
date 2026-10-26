"""Power-capacity additions for storage technologies."""

from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.expression import GenericExpression


class StoragePowerCapacityAddition(GenericExpression):
    """Select storage power-capacity additions."""

    name = "storage_power_capacity_addition"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct select storage power-capacity additions.

        Summary:
            Select storage power-capacity additions.

        Formulation:

        .. math::

            \\Delta K_{h,power,n,y}

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
        return (
            optimization_model.variables[CapacityAddition]
            .sel(
                set_technologies=optimization_model.sets[SetStorageTechnologies],
                set_location=optimization_model.sets[SetNodes],
            )
            .rename(
                {
                    "set_technologies": "set_storage_technologies",
                    "set_location": "set_nodes",
                }
            )
        )
