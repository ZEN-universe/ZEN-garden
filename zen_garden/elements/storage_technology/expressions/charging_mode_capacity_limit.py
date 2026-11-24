"""Storage capacity available while the charging binary is active."""

from zen_garden.elements.storage_technology.expressions.storage_power_capacity_limit import (  # noqa: E501
    StoragePowerCapacityLimit,
)
from zen_garden.elements.storage_technology.variables.charge_storage_binary import (
    ChargeStorageBinary,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class ChargingModeCapacityLimit(GenericExpression):
    """Multiply storage power limit by the charging-mode binary."""

    name = "charging_mode_capacity_limit"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (StoragePowerCapacityLimit,)

    @classmethod
    def should_construct(cls, model_constructor):
        return model_constructor.config.system.storage_charge_discharge_binary and bool(
            model_constructor.optimization_model.sets[SetStorageTechnologies]
        )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct multiply storage power limit by the charging-mode binary.

        Summary:
            Multiply storage power limit by the charging-mode binary.

        Formulation:

        .. math::

            \\bar k^{power}_{h,n,y}B^{ch}_{h,n,y,r}

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
            optimization_model.variables[ChargeStorageBinary]
            * optimization_model.expressions[StoragePowerCapacityLimit]
        )
