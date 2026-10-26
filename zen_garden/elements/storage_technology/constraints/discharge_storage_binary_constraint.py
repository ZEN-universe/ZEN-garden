from zen_garden.elements.storage_technology.expressions.charging_mode_capacity_limit import (  # noqa: E501
    ChargingModeCapacityLimit,
)
from zen_garden.elements.storage_technology.expressions.storage_power_capacity_limit import (  # noqa: E501
    StoragePowerCapacityLimit,
)
from zen_garden.elements.storage_technology.variables.flow_storage_discharge import (
    FlowStorageDischarge,
)
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class DischargeStorageBinaryConstraint(GenericConstraint):
    """Allow discharging only while the storage charging mode is inactive."""

    name = "constraint_discharge_storage_binary"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    masks = (PowerOperationMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return model_constructor.config.system.storage_charge_discharge_binary and bool(
            model_constructor.optimization_model.sets[SetStorageTechnologies]
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct allow discharging only while the storage charging mode is inactive.

        Summary:
            Allow discharging only while the storage charging mode is inactive.

        Formulation:

        .. math::

            F^{dis}+\\bar k^{power}B^{ch}\\leq\\bar k^{power}

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
        return (
            model.variables[FlowStorageDischarge]
            + model.expressions[ChargingModeCapacityLimit]
            <= model.expressions[StoragePowerCapacityLimit]
        )
