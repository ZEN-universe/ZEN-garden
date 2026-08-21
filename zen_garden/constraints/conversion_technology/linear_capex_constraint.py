import linopy as lp
import numpy as np
from linopy.expressions import LinearExpression

from zen_garden.constraints.generic_constraint import GenericConstraint


class LinearCapexConstraint(GenericConstraint):
    def build(self):
        """Summary:
        If capacity and capex have a linear relationship.

        Formulation:

        .. math::
            C^{\\mathrm{cap,overnight}}_{h,n,y} =
            \\kappa^{\\mathrm{cap}}_{h,y} \\Delta K_{h,n,y}

        Notation:

        :math:`C^{\\mathrm{cap,overnight}}_{h,n,y}`: overnight CAPEX of conversion
        technology :math:`h` at node :math:`n` in year :math:`y`
        :math:`\\kappa^{\\mathrm{cap}}_{h,y}`: specific CAPEX
        :math:`\\Delta K_{h,n,y}`: power-capacity addition
        """
        techs = self.zen_model.sets["set_conversion_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        capex_specific_conversion = self.zen_model.parameters.capex_specific_conversion
        capex_specific_conversion = capex_specific_conversion.rename(
            {
                old: new
                for old, new in zip(
                    list(capex_specific_conversion.dims),
                    [
                        "set_technologies",
                        "set_location",
                        "set_years",
                    ],
                    strict=False,
                )
            }
        )

        capacity_addition = self.zen_model.variables["capacity_addition"].loc[
            techs, "power", nodes
        ]
        cost_capex_overnight = self.zen_model.variables["cost_capex_overnight"].loc[
            techs, "power", nodes
        ]

        capex_specific_conversion = capex_specific_conversion.broadcast_like(
            capacity_addition.lower
        )
        mask = ~np.isnan(capex_specific_conversion)
        lhs = lp.merge(
            [
                1 * cost_capex_overnight,
                -capex_specific_conversion * capacity_addition,
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs = self.align_and_mask(lhs, mask)
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_linear_capex", constraints)
