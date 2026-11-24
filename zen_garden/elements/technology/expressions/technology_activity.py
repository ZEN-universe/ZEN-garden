"""Unified operational activity for all technology types."""

import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.storage_technology.expressions.storage_throughput import (
    StorageThroughput,
)
from zen_garden.elements.transport_technology.variables.flow_transport import (
    FlowTransport,
)
from zen_garden.model.component_types.expression import GenericExpression


class TechnologyActivity(GenericExpression):
    """Combine conversion, storage, and transport operational activity."""

    name = "technology_activity"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (ConversionReferenceFlow, StorageThroughput)

    @classmethod
    def should_construct(cls, model_constructor):
        technologies = model_constructor.optimization_model.sets.on_off_technologies()
        return bool(technologies)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct combine conversion, storage, and transport operational activity.

        Summary:
            Combine conversion, storage, and transport operational activity.

        Formulation:

        .. math::

            \\begin{cases}F^{ref}_{h,p,y,r}&h \\in \\mathcal
            H^{conv}\\\\F^{trans}_{h,p,y,r}&h \\in \\mathcal
            H^{trans}\\\\F^{ch}_{h,p,y,r} +F^{dis}_{h,p,y,r}&h \\in \\mathcal
            H^{storage}\\end{cases}

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
        conversion = optimization_model.expressions[ConversionReferenceFlow].rename(
            {
                "set_conversion_technologies": "set_technologies",
                "set_nodes": "set_location",
            }
        )
        transport = optimization_model.variables[FlowTransport].rename(
            {
                "set_transport_technologies": "set_technologies",
                "set_edges": "set_location",
            }
        )
        activities = [conversion, 1 * transport]
        if StorageThroughput in optimization_model.expressions:
            activities.append(optimization_model.expressions[StorageThroughput])
        return lp.merge(
            activities,
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
