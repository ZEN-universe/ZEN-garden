from zen_garden.elements.technology.expressions.annual_technology_emissions import (
    AnnualTechnologyEmissions,
)
from zen_garden.model.component_types.expression import GenericExpression


class CarbonEmissionsTechnologyTotal(GenericExpression):
    """Total annual technology operating emissions."""

    name = "carbon_emissions_technology_total"
    doc = r"""Total annual technology operating emissions.

    .. math::

        \begin{aligned}
        &\sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}\\
        +{}&\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{\mathrm{op}}_h
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})\\
        +{}&\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{trans}}_{h,e,y,r}.
        \end{aligned}
    """
    unit_category = {"emissions": 1}
    indices = ("set_years",)
    dependencies = (AnnualTechnologyEmissions,)

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.expressions[
            AnnualTechnologyEmissions
        ]
