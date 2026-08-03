from zen_garden.constraints.generic_constraint import GenericConstraint


class CostCapexYearlyTotalConstraint(GenericConstraint):
    def build(self):
        """Sums over all technologies to calculate total capex.

        .. math::
            CAPEX_y = \\sum_{h\\in\\mathcal{H}}\\sum_{p\\in\\mathcal{P}}A_{h,p,y} +
            \\sum_{k\\in\\mathcal{K}}\\sum_{n\\in\\mathcal{N}}A^\\mathrm{e}_{k,n,y}

        :math:`A_{h,p,y}`: annual capex of technology :math:`h` at location :math:`p`
        in year :math:`y`

        """
        lhs = self.zen_model.variables[
            "cost_capex_yearly_total"
        ] - self.zen_model.variables["cost_capex_yearly"].sum(
            ["set_technologies", "set_capacity_types", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_capex_yearly_total", constraints)
