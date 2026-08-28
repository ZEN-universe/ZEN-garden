import linopy as lp
import pandas as pd
from linopy.expressions import LinearExpression

from zen_garden.elements.technology.constraints.technology_constraint import (
    TechnologyConstraint,
)
from zen_garden.model.registries.multi_index_helper import MultiIndexHelper


class CostCapexYearlyConstraint(TechnologyConstraint):
    @classmethod
    def build(cls, model_constructor):
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
        """
        index_values, index_names = model_constructor.zen_model.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ]
        )
        index = MultiIndexHelper(index_values, index_names)
        ### masks
        # not needed

        # Annuity factor
        dr = model_constructor.zen_model.parameters.discount_rate
        lt = model_constructor.zen_model.parameters.depreciation_time

        if dr != 0:
            a = ((1 + dr) ** lt * dr) / ((1 + dr) ** lt - 1)
        else:
            a = 1 / lt

        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in index.get_unique(["set_technologies", "set_years"])
                for py in list(
                    cls.get_lifetime_range(
                        model_constructor, t, y, use_depreciation_time=True
                    )
                )
            ]
        )

        lt_range = pd.Series(index=lt_range, data=-1)
        lt_range.index.names = [
            "set_technologies",
            "set_years",
            "set_years_prev",
        ]
        lt_range = (
            lt_range.to_xarray()
            .broadcast_like(model_constructor.zen_model.variables["capacity"].lower)
            .fillna(0)
        )

        cost_capex_overnight = model_constructor.zen_model.variables[
            "cost_capex_overnight"
        ].rename({"set_years": "set_years_prev"})
        cost_capex_overnight = cost_capex_overnight.broadcast_like(lt_range)
        expr = (lt_range * a * cost_capex_overnight).sum("set_years_prev")
        lhs = lp.merge(
            [1 * model_constructor.zen_model.variables["cost_capex_yearly"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = (
            a * model_constructor.zen_model.parameters.existing_capex
        ).broadcast_like(lhs.const)
        constraints = lhs == rhs

        ### return
        model_constructor.zen_model.add_constraint(
            "constraint_cost_capex_yearly", constraints
        )
