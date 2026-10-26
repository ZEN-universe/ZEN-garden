"""Net carrier supply at each model node."""

import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.carrier.expressions.nodal_conversion_consumption import (
    NodalConversionConsumption,
)
from zen_garden.elements.carrier.expressions.nodal_conversion_production import (
    NodalConversionProduction,
)
from zen_garden.elements.carrier.expressions.nodal_storage_charging import (
    NodalStorageCharging,
)
from zen_garden.elements.carrier.expressions.nodal_storage_discharging import (
    NodalStorageDischarging,
)
from zen_garden.elements.carrier.expressions.nodal_transport_inflow import (
    NodalTransportInflow,
)
from zen_garden.elements.carrier.expressions.nodal_transport_outflow import (
    NodalTransportOutflow,
)
from zen_garden.elements.carrier.variables.flow_export import FlowExport
from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.elements.carrier.variables.shed_demand import ShedDemand
from zen_garden.model.component_types.expression import GenericExpression


class NodalNetSupply(GenericExpression):
    """Combine all production, consumption, trade, and storage flows."""

    name = "nodal_net_supply"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (
        NodalConversionConsumption,
        NodalConversionProduction,
        NodalTransportInflow,
        NodalTransportOutflow,
        NodalStorageCharging,
        NodalStorageDischarging,
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct combine all production, consumption, trade, and storage flows.

        Summary:
            Combine all production, consumption, trade, and storage flows.

        Formulation:

        .. math::

             \\sum_{h:c \\in \\mathcal C_h^{out}}F^{out}_{h,c,n,y,r}- \\sum_{h:c \\in
            \\mathcal C_h^{in}}F^{in}_{h,c,n,y,r} + \\sum_{h,e
            \\in\\delta^-(n)}(F^{trans}_{h,e,y,r}-F^{loss}_{h,e,y,r})- \\sum_{h,e
            \\in\\delta^ +(n)}F^{trans}_{h,e,y,r}- \\sum_{h:c_h^{ref}
            =c}F^{ch}_{h,n,y,r} + \\sum_{h:c_h^{ref} =c}F^{dis}_{h,n,y,r}
            +F^{imp}_{c,n,y,r}-F^{exp}_{c,n,y,r} +F^{shed}_{c,n,y,r}

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
        expressions = optimization_model.expressions
        return lp.merge(
            [
                expressions[NodalConversionProduction],
                -expressions[NodalConversionConsumption],
                expressions[NodalTransportInflow],
                -expressions[NodalTransportOutflow],
                -expressions[NodalStorageCharging],
                expressions[NodalStorageDischarging],
                1 * optimization_model.variables[FlowImport],
                -optimization_model.variables[FlowExport],
                1 * optimization_model.variables[ShedDemand],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
