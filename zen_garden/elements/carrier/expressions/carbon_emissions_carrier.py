from zen_garden.elements.carrier.parameters.carbon_intensity_carrier_export import (
    CarbonIntensityCarrierExport,
)
from zen_garden.elements.carrier.parameters.carbon_intensity_carrier_import import (
    CarbonIntensityCarrierImport,
)
from zen_garden.elements.carrier.variables.flow_export import FlowExport
from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.model.component_types.expression import GenericExpression


class CarbonEmissionsCarrier(GenericExpression):
    """Hourly carrier emissions from imports minus exported emissions."""

    name = "carbon_emissions_carrier"
    doc = r"""Hourly emissions from carrier imports and exports.

    .. math::

        \varepsilon^{\mathrm{imp}}_{c,y}F^{\mathrm{imp}}_{c,n,t}
        -\varepsilon^{\mathrm{exp}}_{c,y}F^{\mathrm{exp}}_{c,n,t}
    """
    unit_category = {"emissions": 1, "time": -1}
    indices = ("set_carriers", "set_nodes", "set_years", "set_representative_hours")

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return (
            model.variables[FlowImport] * model.parameters[CarbonIntensityCarrierImport]
            - model.variables[FlowExport]
            * model.parameters[CarbonIntensityCarrierExport]
        )
