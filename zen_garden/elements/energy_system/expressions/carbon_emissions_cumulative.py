import xarray as xr

from zen_garden.elements.energy_system.expressions.carbon_emissions_annual import (
    CarbonEmissionsAnnual,
)
from zen_garden.elements.energy_system.parameters.carbon_emissions_cumulative_existing import (  # noqa: E501
    CarbonEmissionsCumulativeExisting,
)
from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.model.component_types.expression import GenericExpression


class CarbonEmissionsCumulative(GenericExpression):
    """Cumulative emissions before and within the modeled horizon."""

    name = "carbon_emissions_cumulative"
    doc = r"""Cumulative emissions through the current planning period.

    .. math::

        m^{\mathrm{cum}}_0
        +\underbrace{\sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{\mathrm{op}}_h
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})
        +\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{trans}}_{h,e,y,r}
        +\sum_{c,n,r}w_r(
        \varepsilon^{\mathrm{imp}}_{c,y}F^{\mathrm{imp}}_{c,n,y,r}
        -\varepsilon^{\mathrm{exp}}_{c,y}F^{\mathrm{exp}}_{c,n,y,r})}
        _{\text{current-period emissions}}
        +\Delta y\sum_{v<y}\left[
        \sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,v,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{\mathrm{op}}_h
        (F^{\mathrm{charge}}_{h,n,v,r}+F^{\mathrm{discharge}}_{h,n,v,r})
        +\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{trans}}_{h,e,v,r}
        +\sum_{c,n,r}w_r(
        \varepsilon^{\mathrm{imp}}_{c,v}F^{\mathrm{imp}}_{c,n,v,r}
        -\varepsilon^{\mathrm{exp}}_{c,v}F^{\mathrm{exp}}_{c,n,v,r})\right].
    """
    unit_category = {"emissions": 1}
    indices = ("set_years",)
    dependencies = (CarbonEmissionsAnnual,)

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        years = model_constructor.model_schema.set_years
        annual = model.expressions[CarbonEmissionsAnnual]
        preceding_weights = xr.DataArray(
            [
                [
                    model.parameters[IntervalBetweenYears] if previous < year else 0
                    for previous in years
                ]
                for year in years
            ],
            coords={"set_years": years, "set_years_previous": years},
            dims=("set_years", "set_years_previous"),
        )
        preceding = (
            annual.rename(set_years="set_years_previous") * preceding_weights
        ).sum("set_years_previous")
        return model.parameters[CarbonEmissionsCumulativeExisting] + annual + preceding
