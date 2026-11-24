from zen_garden.elements.energy_system.expressions.carbon_emissions_annual import (
    CarbonEmissionsAnnual,
)
from zen_garden.elements.energy_system.parameters.carbon_emissions_annual_limit import (
    CarbonEmissionsAnnualLimit,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_annual_overshoot import (  # noqa: E501
    CarbonEmissionsAnnualOvershoot,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsAnnualLimitConstraint(GenericConstraint):
    """Carbon Emissions Annual Limit."""

    name = "constraint_carbon_emissions_annual_limit"
    doc = r"""Limit annual emissions using only variable and parameter symbols.

    .. math::

        \begin{aligned}
        &\sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{op}_h
        F^{out}_{h,c_h^{ref},n,y,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{op}_h
        (F^{charge}_{h,n,y,r}+F^{discharge}_{h,n,y,r})\\
        &+\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{op}_h
        F^{trans}_{h,e,y,r}
        +\sum_{c,n,r}w_r(\varepsilon^{imp}_{c,y}F^{imp}_{c,n,y,r}
        -\varepsilon^{exp}_{c,y}F^{exp}_{c,n,y,r})
        -M^{ann,over}_y\leq\overline m_y.
        \end{aligned}
    """
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Time dependent carbon emissions limit from technologies and carriers.

        Formulation:

        .. math::
            M_y-M_y^{\\mathrm{ann,over}}\\leq \\overline{m}_y

        Notation:

        :math:`M_y^{\\mathrm{ann,over}}`: permitted overshoot of the annual emissions
        limit

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        lhs = (
            model_constructor.optimization_model.expressions[CarbonEmissionsAnnual]
            - model_constructor.optimization_model.variables[
                CarbonEmissionsAnnualOvershoot
            ]
        )
        rhs = model_constructor.optimization_model.parameters[
            CarbonEmissionsAnnualLimit
        ]
        constraints = lhs <= rhs

        return constraints
