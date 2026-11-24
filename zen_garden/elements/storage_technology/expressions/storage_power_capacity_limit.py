"""Power-capacity limit for storage operation."""

from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.parameters.capacity_limit import CapacityLimit
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class StoragePowerCapacityLimit(GenericExpression):
    """Select and align storage power-capacity limits."""

    name = "storage_power_capacity_limit"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct select and align storage power-capacity limits.

        Summary:
            Select and align storage power-capacity limits.

        Formulation:

        .. math::

            \\bar k_{h,power,n,y}

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
            optimization_model.parameters[CapacityLimit]
            .rename(
                {
                    "set_technologies": "set_storage_technologies",
                    "set_location": "set_nodes",
                }
            )
            .sel(
                set_storage_technologies=optimization_model.sets[
                    SetStorageTechnologies
                ],
                set_nodes=optimization_model.sets[SetNodes],
            )
        )
