"""Dependent-carrier flow of conversion technologies."""

import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.conversion_technology.sets.relation_conversion_technology_dependent_carrier import (  # noqa: E501
    RelationConversionTechnologyDependentCarrier,
)
from zen_garden.elements.conversion_technology.variables.flow_conversion_input import (
    FlowConversionInput,
)
from zen_garden.elements.conversion_technology.variables.flow_conversion_output import (
    FlowConversionOutput,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.utils import align_like


class ConversionDependentFlow(GenericExpression):
    """Select all configured dependent-carrier conversion flows."""

    name = "conversion_dependent_flow"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_conversion_technologies",
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct select all configured dependent-carrier conversion flows.

        Summary:
            Select all configured dependent-carrier conversion flows.

        Formulation:

        .. math::

            \\{F^{in}_{h,c,n,y,r}:c \\in \\mathcal C_h^{dep} \\cap \\mathcal C_h^{in}\\}
            \\cup\\{F^{out}_{h,c,n,y,r}:c \\in \\mathcal C_h^{dep} \\cap \\mathcal
            C_h^{out}\\}

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
        input_flow = optimization_model.variables[FlowConversionInput]
        output_flow = optimization_model.variables[FlowConversionOutput]
        dependent_mask = optimization_model.sets.relation_mask(
            RelationConversionTechnologyDependentCarrier
        )
        input_mask = align_like(
            dependent_mask,
            input_flow,
            astype=bool,
        )
        output_mask = align_like(
            dependent_mask,
            output_flow,
            astype=bool,
        )
        return lp.merge(
            [1 * input_flow.where(input_mask), 1 * output_flow.where(output_mask)],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
