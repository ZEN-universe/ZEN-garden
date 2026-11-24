"""Nodal carrier consumption from storage charging."""

import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.elements.energy_system.sets.set_carriers import SetCarriers
from zen_garden.elements.storage_technology.variables.flow_storage_charge import (
    FlowStorageCharge,
)
from zen_garden.elements.technology.sets.relation_technology_reference_carrier import (  # noqa: E501
    RelationTechnologyReferenceCarrier,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class NodalStorageCharging(GenericExpression):
    """Aggregate charging flow by storage reference carrier and node."""

    name = "nodal_storage_charging"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct aggregate charging flow by storage reference carrier and node.

        Summary:
            Aggregate charging flow by storage reference carrier and node.

        Formulation:

        .. math::

            \\sum_{h:c_h^{ref}=c}F^{ch}_{h,n,y,r}

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
        variable = optimization_model.variables[FlowStorageCharge]
        if variable.size == 0:
            return optimization_model.variables[FlowImport].where(False).to_linexpr()
        carriers = list(optimization_model.sets[SetCarriers])
        flows = []
        for carrier in carriers:
            technologies = optimization_model.sets.project_relation(
                RelationTechnologyReferenceCarrier,
                SetStorageTechnologies,
                set_carriers=carrier,
            )
            flows.append(variable.loc[technologies].sum("set_storage_technologies"))
        merged = lp.merge(flows, dim="group", join="outer", cls=LinearExpression)
        return optimization_model.constraints.reorder_group(
            merged,
            None,
            None,
            carriers,
            [SetCarriers.name],
            optimization_model.lp_model,
        )
