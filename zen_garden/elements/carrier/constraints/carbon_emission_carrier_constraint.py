from zen_garden.elements.carrier.parameters.carbon_intensity_carrier_export import (
    CarbonIntensityCarrierExport,
)
from zen_garden.elements.carrier.parameters.carbon_intensity_carrier_import import (
    CarbonIntensityCarrierImport,
)
from zen_garden.elements.carrier.variables.carbon_emissions_carrier import (
    CarbonEmissionsCarrier,
)
from zen_garden.elements.carrier.variables.flow_export import FlowExport
from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsCarrierConstraint(GenericConstraint):
    """Carbon Emissions Carrier."""

    name = "constraint_carbon_emissions_carrier"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Carbon emissions of importing and exporting carrier.

        Formulation:

        .. math::
           M^{\\mathrm{carrier}}_{c,n,t} = \\varepsilon^{\\mathrm{imp}}_{c,y}
           F^{\\mathrm{imp}}_{c,n,t} - \\varepsilon^{\\mathrm{exp}}_{c,y}
           F^{\\mathrm{exp}}_{c,n,t}

        Notation:

        :math:`M^{\\mathrm{carrier}}_{c,n,t}`: carbon emissions of importing
        and exporting carrier :math:`c` at node :math:`n` in time step :math:`t`
        of year :math:`y`
        :math:`\\varepsilon^{\\mathrm{imp}}_{c,y}`: carbon intensity of carrier import
        :math:`c` in year :math:`y`
        :math:`\\varepsilon^{\\mathrm{exp}}_{c,y}`: carbon intensity of carrier export
        :math:`c` in year :math:`y`
        :math:`F^{\\mathrm{imp}}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{exp}}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` in time step :math:`t` of year :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        carbon_intensity_carrier_import = (
            model_constructor.optimization_model.parameters[
                CarbonIntensityCarrierImport
            ]
        )
        carbon_intensity_carrier_export = (
            model_constructor.optimization_model.parameters[
                CarbonIntensityCarrierExport
            ]
        )
        lhs = model_constructor.optimization_model.variables[CarbonEmissionsCarrier] - (
            model_constructor.optimization_model.variables[FlowImport]
            * carbon_intensity_carrier_import
            - model_constructor.optimization_model.variables[FlowExport]
            * carbon_intensity_carrier_export
        )

        rhs = 0

        constraints = lhs == rhs

        return constraints
