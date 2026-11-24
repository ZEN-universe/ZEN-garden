import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.storage_technology.expressions.storage_throughput import (
    StorageThroughput,
)
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.parameters.opex_specific_variable import (
    OpexSpecificVariable,
)
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.transport_technology.variables.flow_transport import (
    FlowTransport,
)
from zen_garden.model.component_types.expression import GenericExpression


class CostOpexVariable(GenericExpression):
    """Hourly variable operating cost for every technology type."""

    name = "cost_opex_variable"
    doc = r"""Hourly variable operating cost, expanded by technology type.

    .. math::

        \begin{cases}
        \kappa^{\mathrm{op,var}}_{h,y}
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,t},&h\in\mathcal H^{conv},\\
        \kappa^{\mathrm{op,var}}_{h,y}
        (F^{\mathrm{charge}}_{h,n,t}+F^{\mathrm{discharge}}_{h,n,t}),
        &h\in\mathcal H^{stor},\\
        \kappa^{\mathrm{op,var}}_{h,y}F^{\mathrm{trans}}_{h,e,t},
        &h\in\mathcal H^{trans}.
        \end{cases}
    """
    unit_category = {"money": 1, "time": -1}
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (ConversionReferenceFlow, StorageThroughput)
    masks = (PowerOperationMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        pieces = []
        nodes = model.sets[SetNodes]
        conversion = model.sets[SetConversionTechnologies]
        if conversion:
            factor = model.parameters[OpexSpecificVariable].sel(
                set_technologies=conversion, set_location=nodes
            )
            pieces.append(
                (
                    factor
                    * model.expressions[ConversionReferenceFlow].rename(
                        set_conversion_technologies="set_technologies",
                        set_nodes="set_location",
                    )
                )
            )
        storage = model.sets[SetStorageTechnologies]
        if storage:
            factor = model.parameters[OpexSpecificVariable].sel(
                set_technologies=storage, set_location=nodes
            )
            pieces.append(factor * model.expressions[StorageThroughput])
        transport = model.sets[SetTransportTechnologies]
        if transport:
            factor = model.parameters[OpexSpecificVariable].sel(
                set_technologies=transport
            )
            pieces.append(
                factor
                * model.variables[FlowTransport].rename(
                    set_transport_technologies="set_technologies",
                    set_edges="set_location",
                )
            )
        return lp.merge(
            pieces, compat="broadcast_equals", join="outer", cls=LinearExpression
        )
