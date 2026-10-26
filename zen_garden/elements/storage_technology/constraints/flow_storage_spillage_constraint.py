from zen_garden.elements.storage_technology.masks import StorageOperationMask
from zen_garden.elements.storage_technology.parameters.flow_storage_inflow import (
    FlowStorageInflow,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class FlowStorageSpillageConstraint(GenericConstraint):
    """Flow Storage Spillage."""

    name = "constraint_flow_storage_spillage"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    masks = (StorageOperationMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Limit storage spillage to the exogenous storage inflow.

        Formulation:

        .. math::
            0 \\leq F^{\\mathrm{spill}}_{h,n,t} \\leq q^{\\mathrm{in}}_{h,n,t}

        The lower bound is imposed by the nonnegative variable definition.

        Notation:

        :math:`F^{\\mathrm{spill}}_{h,n,t}`: storage spillage of storage technology
        :math:`h`
        at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`q^{\\mathrm{in}}_{h,n,t}`: exogenous inflow into storage technology
        :math:`h` at node :math:`n` in time step :math:`t` of year :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        flow_storage_inflow = model_constructor.optimization_model.parameters[
            FlowStorageInflow
        ]
        flow_storage_spillage = (
            model_constructor.optimization_model.lp_model.variables.flow_storage_spillage
        )

        lhs = flow_storage_spillage - flow_storage_inflow
        rhs = 0
        constraints = lhs <= rhs

        return constraints
