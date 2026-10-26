from zen_garden.elements.storage_technology.expressions.charging_mode_capacity_limit import (  # noqa: E501
    ChargingModeCapacityLimit,
)
from zen_garden.elements.storage_technology.variables.flow_storage_charge import (
    FlowStorageCharge,
)
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class ChargeStorageBinaryConstraint(GenericConstraint):
    """Allow charging only while the storage charging mode is active."""

    name = "constraint_charge_storage_binary"
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
        """Construct allow charging only while the storage charging mode is active.

        Summary:
            Allow charging only while the storage charging mode is active.

        Formulation:

        .. math::

            F^{ch}\\leq\\bar k^{power}B^{ch}

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
            model.variables[FlowStorageCharge]
            <= model.expressions[ChargingModeCapacityLimit]
        )
