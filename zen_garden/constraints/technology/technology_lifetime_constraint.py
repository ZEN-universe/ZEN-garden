import itertools

import linopy as lp
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.constraints.technology.technology_constraint import TechnologyConstraint


class TechnologyLifetimeConstraint(TechnologyConstraint):
    def build(self):
        """Calculates remaining capacity of technologies based on the lifetime.

        limited lifetime of the technologies. calculates 'capacity', i.e., the
        capacity at the end of the year and 'capacity_previous', i.e., the capacity at
        the beginning of the year.

        .. math::
            S_{h,p,y} = \\sum_{\\tilde{y}=\\max(y_0,y-\\lceil\\frac{l_h}
            {\\Delta^\\mathrm{y}}\\rceil+1)}^y \\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}
            {\\Delta^\\mathrm{y}}\\rceil+1))}^{\\psi(y_0)}
            \\Delta s^\\mathrm{ex}_{h,p,\\hat{y}}

        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested capacity
        after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex}_{h,p,y}`: size of the previously invested
        capacities at location :math:`p` in year :math:`y`
        """
        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in itertools.product(
                    self.zen_model.sets["set_technologies"],
                    self.zen_model.sets["set_years"],
                )
                for py in list(self.get_lifetime_range(t, y))
            ],
            names=[
                "set_technologies",
                "set_years",
                "set_years_prev",
            ],
        )
        lt_range = pd.Series(index=lt_range, data=-1)
        lt_range = (
            lt_range.to_xarray()
            .broadcast_like(self.zen_model.variables["capacity"].lower)
            .fillna(0)
        )
        capacity_addition = self.zen_model.variables["capacity_addition"].rename(
            {"set_years": "set_years_prev"}
        )
        capacity_addition = capacity_addition.broadcast_like(lt_range)
        expr = (lt_range * capacity_addition).sum("set_years_prev")
        lhs = lp.merge(
            [1 * self.zen_model.variables["capacity"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs_previous = lp.merge(
            [
                1 * self.zen_model.variables["capacity_previous"],
                expr,
                1 * self.zen_model.variables["capacity_addition"],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const, self.zen_model.parameters.existing_capacities, join="left"
        )[1]
        constraints = lhs == rhs
        constraints_previous = lhs_previous == rhs

        ### return
        self.zen_model.add_constraint("constraint_technology_lifetime", constraints)
        self.zen_model.add_constraint(
            "constraint_technology_lifetime_previous", constraints_previous
        )
