import linopy as lp
import pandas as pd
from linopy.expressions import LinearExpression

from zen_garden.elements.technology.constraints.technology_constraint import TechnologyConstraint
from zen_garden.elements.technology import Technology
from zen_garden.model.components.multi_index_helper import MultiIndexHelper


class CostCapexYearlyConstraint(TechnologyConstraint):
    def build(self):
        """Summary:
        Aggregates the capex of built capacity and of existing capacity.

        Formulation:

        .. math::
            C^{\\mathrm{cap,ann}}_{h,p,y} = a^{\\mathrm{ann}}_h\\left(
            \\sum_{\\tilde y\\in\\mathcal{Y}^{\\mathrm{dep}}_{h,y}}
            C^{\\mathrm{cap,overnight}}_{h,p,\\tilde y} +
            \\kappa^{\\mathrm{cap,ex}}_{h,p,y}\\right)

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
        index_values, index_names = self.zen_model.create_custom_set(
            [
                "set_technologies",
                "set_capacity_types",
                "set_location",
                "set_years",
            ],
            Technology,
        )
        index = MultiIndexHelper(index_values, index_names)
        ### masks
        # not needed

        # Annuity factor
        dr = self.zen_model.parameters.discount_rate
        lt = self.zen_model.parameters.depreciation_time

        if dr != 0:
            a = ((1 + dr) ** lt * dr) / ((1 + dr) ** lt - 1)
        else:
            a = 1 / lt

        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in index.get_unique(["set_technologies", "set_years"])
                for py in list(
                    self.get_lifetime_range(t, y, use_depreciation_time=True)
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
            .broadcast_like(self.zen_model.variables["capacity"].lower)
            .fillna(0)
        )

        cost_capex_overnight = self.zen_model.variables["cost_capex_overnight"].rename(
            {"set_years": "set_years_prev"}
        )
        cost_capex_overnight = cost_capex_overnight.broadcast_like(lt_range)
        expr = (lt_range * a * cost_capex_overnight).sum("set_years_prev")
        lhs = lp.merge(
            [1 * self.zen_model.variables["cost_capex_yearly"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = (a * self.zen_model.parameters.existing_capex).broadcast_like(lhs.const)
        constraints = lhs == rhs

        ### return
        self.zen_model.add_constraint("constraint_cost_capex_yearly", constraints)
