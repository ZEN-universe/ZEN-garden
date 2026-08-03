from zen_garden.constraints.generic_constraint import GenericConstraint


class CostOpexYearlyTotalConstraint(GenericConstraint):
    def build(self):
        """Sums over all technologies to calculate total opex.

        .. math::
            OPEX_y = \\sum_{h\\in\\mathcal{H}}\\sum_{p\\in\\mathcal{P}} OPEX_{h,p,y}

        :math:`OPEX_{h,p,y}`: opex of operating technology :math:`h` at
        location :math:`p` in year :math:`y`

        """
        lhs = self.zen_model.variables[
            "cost_opex_yearly_total"
        ] - self.zen_model.variables["cost_opex_yearly"].sum(
            ["set_technologies", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_opex_yearly_total", constraints)
