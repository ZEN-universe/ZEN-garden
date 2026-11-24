from zen_garden.elements.carrier.expressions.annual_import_export_emissions import (
    AnnualImportExportEmissions,
)
from zen_garden.model.component_types.expression import GenericExpression


class CarbonEmissionsCarrierTotal(GenericExpression):
    """Annual emissions caused by carrier imports and exports."""

    name = "carbon_emissions_carrier_total"
    doc = r"""Annual emissions from carrier imports and exports.

    .. math::

        \sum_{c,n,r}w_r\left(
        \varepsilon^{\mathrm{imp}}_{c,y}F^{\mathrm{imp}}_{c,n,y,r}
        -\varepsilon^{\mathrm{exp}}_{c,y}F^{\mathrm{exp}}_{c,n,y,r}
        \right)
    """
    unit_category = {"emissions": 1}
    indices = ("set_years",)
    dependencies = (AnnualImportExportEmissions,)

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.expressions[
            AnnualImportExportEmissions
        ]
