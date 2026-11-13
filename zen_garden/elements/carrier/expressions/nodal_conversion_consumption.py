"""Nodal carrier consumption by conversion technologies."""

import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.conversion_technology.variables.flow_conversion_input import (
    FlowConversionInput,
)
from zen_garden.elements.energy_system.sets.set_carriers import SetCarriers
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class NodalConversionConsumption(GenericExpression):
    """Aggregate conversion input flow by carrier and node."""

    name = "nodal_conversion_consumption"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct aggregate conversion input flow by carrier and node.

        Summary:
            Aggregate conversion input flow by carrier and node.

        Formulation:

        .. math::

            \\sum_{h:c\\in\\mathcal C_h^{in}}F^{in}_{h,c,n,y,r}

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
            technologies = [
                technology
                for technology in optimization_model.sets[SetConversionTechnologies]
                if carrier
                in optimization_model.sets.get_related(
                    "set_conversion_technologies", technology, "set_input_carriers"
                )
            ]
            selected_carrier = [carrier] if technologies else []
            variable = optimization_model.variables[FlowConversionInput]
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
