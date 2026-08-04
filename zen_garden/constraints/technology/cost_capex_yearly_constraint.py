import linopy as lp
import pandas as pd
from linopy.expressions import LinearExpression

from zen_garden.constraints.technology.technology_constraint import TechnologyConstraint
from zen_garden.model.components.multi_index_helper import MultiIndexHelper


class CostCapexYearlyConstraint(TechnologyConstraint):
    def build(self, index: MultiIndexHelper | None = None):
        """Aggregates the capex of built capacity and of existing capacity.

        .. math::
            A_{h,p,y} = f_h (\\sum_{\\tilde{y} = \\max(y_0,y-\\lceil\\frac{l_h}
            {\\mathrm{dy}}\\rceil+1)}^y \\alpha_{h,y}\\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}
            {\\mathrm{dy}}\\rceil+1))}^{\\psi(y_0)} \\alpha_{h,y_0}
            \\Delta s^\\mathrm{ex}_{h,p,\\hat{y}})

        :math:`A_{h,p,y}`: annual capex of technology :math:`h` at location :math:`p`
        in year :math:`y` \n
        :math:`f_h`: annuity factor of technology :math:`h` \n
        :math:`\\alpha_{h,y}`: unit cost of capital investment of technology :math:`h`
        in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested capacity
        after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex}_{h,p,y}`: size of the previously added capacities
        at location :math:`p` in year :math:`y` \n
        :math:`l_h`: depreciation time of technology :math:`h`   \n
        :math:`\\mathrm{dy}`: interval between planning periods


        """
        assert index is not None, "index must be provided"
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
