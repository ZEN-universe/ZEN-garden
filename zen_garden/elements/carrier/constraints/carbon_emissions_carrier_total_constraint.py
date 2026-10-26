from zen_garden.elements.carrier.expressions.annual_import_export_emissions import (
    AnnualImportExportEmissions,
)
from zen_garden.elements.carrier.variables.carbon_emissions_carrier_total import (
    CarbonEmissionsCarrierTotal,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsCarrierTotalConstraint(GenericConstraint):
    """Carbon Emissions Carrier Total."""

    name = "constraint_carbon_emissions_carrier_total"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Total carbon emissions of importing and exporting carrier.

        Formulation:

        .. math::
            M_y^{\\mathrm{carrier}} = \\sum_{c\\in\\mathcal{C}}\\sum_{n\\in\\mathcal{N}}
            \\sum_{t\\in\\mathcal{T}_y} \\Delta t_t
            M^{\\mathrm{carrier}}_{c,n,t}

        Notation:

        :math:`M^{\\mathrm{carrier}}_{c,n,t}`: carbon emissions of importing
        and exporting carrier :math:`c` at node :math:`n` in time step :math:`t`
        of year :math:`y`
        :math:`\\Delta t_t`: duration of time step :math:`t`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        return (
            model.variables[CarbonEmissionsCarrierTotal]
            == model.expressions[AnnualImportExportEmissions]
        )
