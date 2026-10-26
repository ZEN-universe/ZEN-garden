"""Energy-capacity additions for storage technologies."""

from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class StorageEnergyCapacityAddition(GenericExpression):
    """Select storage energy-capacity additions."""

    name = "storage_energy_capacity_addition"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct select storage energy-capacity additions.

        Summary:
            Select storage energy-capacity additions.

        Formulation:

        .. math::

            \\Delta K_{h,energy,n,y}

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
        return optimization_model.variables[CapacityAdditionEnergy]
