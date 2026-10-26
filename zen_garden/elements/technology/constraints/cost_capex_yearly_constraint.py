import linopy as lp
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.technology.expressions.annualized_capex_from_existing_capacity import (  # noqa: E501
    AnnualizedCapexFromExistingCapacity,
)
from zen_garden.elements.technology.expressions.annualized_capex_from_new_investments import (  # noqa: E501
    AnnualizedCapexFromNewInvestments,
)
from zen_garden.elements.technology.masks import CapexActiveMask
from zen_garden.elements.technology.variables.cost_capex_yearly import CostCapexYearly
from zen_garden.model.component_types.constraint import GenericConstraint


class CostCapexYearlyConstraint(GenericConstraint):
    """Cost Capex Yearly."""

    name = "constraint_cost_capex_yearly"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapexActiveMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Aggregates the capex of built capacity and of existing capacity.

        Formulation:

        .. math::
            C^{\\mathrm{cap,ann}}_{h,p,y} = a^{\\mathrm{ann}}_h\\left(
            \\sum_{\\tilde y\\in\\mathcal{Y}^{\\mathrm{dep}}_{h,y}}
            C^{\\mathrm{cap,overnight}}_{h,p,\\tilde y} +
            \\kappa^{\\mathrm{cap,ex}}_{h,p,y}\\right)

        The annuity factor is

        .. math::
            a^{\\mathrm{ann}}_h =
            \\begin{cases}
            \\dfrac{(1+r^{\\mathrm{disc}})^{L_h^{\\mathrm{dep}}}
            r^{\\mathrm{disc}}}
            {(1+r^{\\mathrm{disc}})^{L_h^{\\mathrm{dep}}}-1},
            & r^{\\mathrm{disc}} \\ne 0, \\\\
            \\dfrac{1}{L_h^{\\mathrm{dep}}}, & r^{\\mathrm{disc}} = 0.
            \\end{cases}

        Storage power- and energy-capacity CAPEX are stored separately and the
        equation is applied to both terms before aggregation.

        Notation:

        :math:`C^{\\mathrm{cap,ann}}_{h,p,y}`: annualized CAPEX for technology
        :math:`h` at location :math:`p` in year :math:`y`
        :math:`a^{\\mathrm{ann}}_h`: annuity factor calculated from discount rate and
        depreciation time
        :math:`\\mathcal{Y}^{\\mathrm{dep}}_{h,y}`: modeled investment years whose
        depreciation period still includes :math:`y`
        :math:`C^{\\mathrm{cap,overnight}}_{h,p,y}`: overnight CAPEX of modeled
        additions
        :math:`\\kappa^{\\mathrm{cap,ex}}_{h,p,y}`: remaining overnight CAPEX of
        existing capacity

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
"""
        model = model_constructor.optimization_model
        lhs = lp.merge(
            [
                1 * model.variables[CostCapexYearly],
                -model.expressions[AnnualizedCapexFromNewInvestments],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const,
            model.expressions[AnnualizedCapexFromExistingCapacity],
            join="left",
        )[1]
        return lhs == rhs
