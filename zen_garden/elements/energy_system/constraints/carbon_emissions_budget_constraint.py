from zen_garden.elements.energy_system.expressions.emissions_counted_toward_budget import (  # noqa: E501
    EmissionsCountedTowardBudget,
)
from zen_garden.elements.energy_system.parameters.carbon_emissions_budget import (
    CarbonEmissionsBudget,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsBudgetConstraint(GenericConstraint):
    """Carbon Emissions Budget."""

    name = "constraint_carbon_emissions_budget"
    doc = r"""Limit emissions counted through the end of the modeled horizon.

    .. math::

        \begin{aligned}
        m^{cum}_0
        +{}&[1+\mathbb 1_{y\ne y_H}(\Delta y-1)]\Bigg\{
        \sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{op}_h
        F^{out}_{h,c_h^{ref},n,y,r}\\
        &+\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{op}_h
        (F^{charge}_{h,n,y,r}+F^{discharge}_{h,n,y,r})
        +\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{op}_h
        F^{trans}_{h,e,y,r}\\
        &+\sum_{c,n,r}w_r(\varepsilon^{imp}_{c,y}F^{imp}_{c,n,y,r}
        -\varepsilon^{exp}_{c,y}F^{exp}_{c,n,y,r})\Bigg\}\\
        +{}&\Delta y\sum_{v<y}\Bigg\{
        \sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{op}_h
        F^{out}_{h,c_h^{ref},n,v,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{op}_h
        (F^{charge}_{h,n,v,r}+F^{discharge}_{h,n,v,r})\\
        &+\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{op}_h
        F^{trans}_{h,e,v,r}
        +\sum_{c,n,r}w_r(\varepsilon^{imp}_{c,v}F^{imp}_{c,n,v,r}
        -\varepsilon^{exp}_{c,v}F^{exp}_{c,n,v,r})\Bigg\}
        -M^{bud,over}_y\leq\overline m^{budget}.
        \end{aligned}
    """
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Carbon emissions budget of whole time horizon.
        The prediction extends until the end of the horizon, i.e., last optimization
        time step plus the current carbon emissions until the end of the horizon.

        Formulation:

        .. math::
            M_y^{\\mathrm{cum}}
            + \\mathbb{1}_{y\\neq y_\\mathrm{H}}(\\Delta y-1)M_y
            - M_y^{\\mathrm{bud,over}} \\leq \\overline{m}^{\\mathrm{budget}}

        Notation:

        :math:`M_y^{\\mathrm{cum}}`: cumulative carbon emissions of energy
        system in year :math:`y`
        :math:`M_y`: annual carbon emissions of energy system in year :math:`y`
        :math:`M_y^{\\mathrm{bud,over}}`: cumulative carbon emissions budget overshoot
        of energy system
        :math:`\\overline{m}^{\\mathrm{budget}}`: carbon emissions budget of energy
        system
        :math:`y_\\mathrm{H}`: final year of the entire optimization horizon. The
        extrapolation term is omitted there because no intermediate years remain.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        return model.expressions[EmissionsCountedTowardBudget] <= (
            model.parameters[CarbonEmissionsBudget]
        )
