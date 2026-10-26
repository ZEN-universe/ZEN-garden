"""Overnight CAPEX of storage energy-capacity additions."""

from zen_garden.elements.storage_technology.parameters.capex_specific_storage_energy import (  # noqa: E501
    CapexSpecificStorageEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class StorageOvernightCapexEnergy(GenericExpression):
    """Multiply storage energy additions by their specific CAPEX."""

    name = "storage_overnight_capex_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct storage energy-capacity investment cost.

        Summary:
            Multiply storage energy-capacity additions by their specific CAPEX.

        Formulation:

        .. math::

            k^{capex,storage,E}_{h,n,y}\Delta E_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy-capacity investment cost.
        """
        model = model_constructor.optimization_model
        return (
            model.parameters[CapexSpecificStorageEnergy]
            * model.variables[CapacityAdditionEnergy]
        )
