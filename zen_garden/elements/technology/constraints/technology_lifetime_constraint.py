import itertools

import linopy as lp
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.technology.constraints.technology_constraint import (
    TechnologyConstraint,
)


class TechnologyLifetimeConstraint(TechnologyConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Calculates remaining capacity of technologies based on the lifetime.

        limited lifetime of the technologies. calculates 'capacity', i.e., the
        capacity at the end of the year and 'capacity_previous', i.e., the capacity at
        the beginning of the year.

        Formulation:

        .. math::
            K_{h,p,y} = k^{\\mathrm{ex}}_{h,p,y}
            + \\sum_{\\tilde y\\in\\mathcal{Y}^{\\mathrm{life}}_{h,y}}
            \\Delta K_{h,p,\\tilde y}

        .. math::
            K^{\\mathrm{prev}}_{h,p,y}
            = K_{h,p,y}-\\Delta K_{h,p,y}

        For storage technologies, each equation is applied independently to power
        and energy capacity.

        Notation:

        :math:`\\mathcal{Y}^{\\mathrm{life}}_{h,y}` contains modeled additions that
        remain active according to
        :math:`\\lfloor L_h/\\Delta y\\rfloor`
        :math:`k^{\\mathrm{ex}}_{h,p,y}`: surviving existing capacity
        :math:`K^{\\mathrm{prev}}_{h,p,y}`: capacity available before the current
        year's addition
        """
        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in itertools.product(
                    model_constructor.zen_model.sets["set_technologies"],
                    model_constructor.zen_model.sets["set_years"],
                )
                for py in list(cls.get_lifetime_range(model_constructor, t, y))
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
            .broadcast_like(model_constructor.zen_model.variables["capacity"].lower)
            .fillna(0)
        )
        capacity_addition = model_constructor.zen_model.variables[
            "capacity_addition"
        ].rename({"set_years": "set_years_prev"})
        capacity_addition = capacity_addition.broadcast_like(lt_range)
        expr = (lt_range * capacity_addition).sum("set_years_prev")
        lhs = lp.merge(
            [1 * model_constructor.zen_model.variables["capacity"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs_previous = lp.merge(
            [
                1 * model_constructor.zen_model.variables["capacity_previous"],
                expr,
                1 * model_constructor.zen_model.variables["capacity_addition"],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const,
            model_constructor.zen_model.parameters.existing_capacities,
            join="left",
        )[1]
        constraints = lhs == rhs
        constraints_previous = lhs_previous == rhs

        ### return
        model_constructor.zen_model.add_constraint(
            "constraint_technology_lifetime", constraints
        )
        model_constructor.zen_model.add_constraint(
            "constraint_technology_lifetime_previous", constraints_previous
        )
