from zen_garden.topology.generic_constraint import GenericConstraint


class CostCapexYearlyTotalConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Sums over all technologies to calculate total capex.

        Formulation:

        .. math::
            C^{\\mathrm{cap}}_y = \\sum_{h\\in\\mathcal{H}}
            \\sum_{p\\in\\mathcal{P}_h}C^{\\mathrm{cap,ann}}_{h,p,y}

        Notation:

        :math:`C^{\\mathrm{cap,ann}}_{h,p,y}`: annual CAPEX of technology :math:`h` at
        location :math:`p` in year :math:`y`
        :math:`p` in year :math:`y`, including all applicable capacity types
        """
        lhs = self.zen_model.variables[
            "cost_capex_yearly_total"
        ] - self.zen_model.variables["cost_capex_yearly"].sum(
            ["set_technologies", "set_capacity_types", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_capex_yearly_total", constraints)
