from zen_garden.elements.carrier.expressions.carbon_emissions_carrier_total import (
    CarbonEmissionsCarrierTotal,
)
from zen_garden.elements.technology.expressions.carbon_emissions_technology_total import (  # noqa: E501
    CarbonEmissionsTechnologyTotal,
)
from zen_garden.model.component_types.expression import GenericExpression


class CarbonEmissionsAnnual(GenericExpression):
    """Annual emissions from technologies and carrier trading."""

    name = "carbon_emissions_annual"
    doc = r"""Annual emissions from technology operation and carrier trading.

    .. math::

        \begin{aligned}
        &\sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}\\
        +{}&\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{\mathrm{op}}_h
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})\\
        +{}&\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{trans}}_{h,e,y,r}\\
        +{}&\sum_{c,n,r}w_r\left(
        \varepsilon^{\mathrm{imp}}_{c,y}F^{\mathrm{imp}}_{c,n,y,r}
        -\varepsilon^{\mathrm{exp}}_{c,y}F^{\mathrm{exp}}_{c,n,y,r}\right).
        \end{aligned}
    """
    unit_category = {"emissions": 1}
    indices = ("set_years",)
    dependencies = (CarbonEmissionsTechnologyTotal, CarbonEmissionsCarrierTotal)

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return (
            model.expressions[CarbonEmissionsTechnologyTotal]
            + model.expressions[CarbonEmissionsCarrierTotal]
        )
