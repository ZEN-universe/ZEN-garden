import linopy as lp
import numpy as np
from linopy.expressions import LinearExpression

from zen_garden.constraints.generic_constraint import GenericConstraint


class LinearCapexConstraint(GenericConstraint):
    def build(self):
        """If capacity and capex have a linear relationship.

        .. math::
            A_{h,p,y}^{approximation} = \\alpha_{h,n,y} \\Delta S_{h,p,y}^{approx}

        :math:`A_{h,p,y}^{approx}`: approximated capex of the technology :math:`h`
        at node :math:`p` in year :math:`y` \n
        :math:`\\alpha_{h,n,y}`: specific capex of the technology :math:`h` at
        node :math:`n` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}^{approx}`: approximated capacity of the
        technology :math:`h` at node :math:`p` in year :math:`y`

        """
        capex_specific_conversion = self.zen_model.parameters.capex_specific_conversion
        capex_specific_conversion = capex_specific_conversion.rename(
            {
                old: new
                for old, new in zip(
                    list(capex_specific_conversion.dims),
                    [
                        "set_conversion_technologies",
                        "set_nodes",
                        "set_years",
                    ],
                    strict=False,
                )
            }
        )
        capex_specific_conversion = capex_specific_conversion.broadcast_like(
            self.zen_model.variables["capacity_approximation"].lower
        )
        mask = ~np.isnan(capex_specific_conversion)
        lhs = lp.merge(
            [
                1 * self.zen_model.variables["capex_approximation"],
                -capex_specific_conversion
                * self.zen_model.variables["capacity_approximation"],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs = self.align_and_mask(lhs, mask)
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_linear_capex", constraints)
