"""Nodal carrier production by conversion technologies."""

import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.conversion_technology.sets.relation_conversion_technology_output_carrier import (  # noqa: E501
    RelationConversionTechnologyOutputCarrier,
)
from zen_garden.elements.conversion_technology.variables.flow_conversion_output import (
    FlowConversionOutput,
)
from zen_garden.elements.energy_system.sets.set_carriers import SetCarriers
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class NodalConversionProduction(GenericExpression):
    """Aggregate conversion output flow by carrier and node."""

    name = "nodal_conversion_production"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct aggregate conversion output flow by carrier and node.

        Summary:
            Aggregate conversion output flow by carrier and node.

        Formulation:

        .. math::

            \\sum_{h:c\\in\\mathcal C_h^{out}}F^{out}_{h,c,n,y,r}

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
        carriers = list(optimization_model.sets[SetCarriers])
        flows = []
        nodes = list(optimization_model.sets[SetNodes])
        for carrier in carriers:
            technologies = optimization_model.sets.project_relation(
                RelationConversionTechnologyOutputCarrier,
                SetConversionTechnologies,
                set_carriers=carrier,
            )
            selected_carrier = [carrier] if len(technologies) else []
            variable = optimization_model.variables[FlowConversionOutput]
            flows.append(
                variable.loc[technologies, selected_carrier, nodes].sum(
                    variable.dims[:2]
                )
            )
        merged = lp.merge(flows, dim="group", join="outer", cls=LinearExpression)
        return optimization_model.constraints.reorder_group(
            merged,
            None,
            None,
            carriers,
            [SetCarriers.name],
            optimization_model.lp_model,
        )
