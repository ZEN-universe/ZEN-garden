from zen_garden.model.component_types.constraint import GenericConstraint


class CostCapexYearlyTotalConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
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
        lhs = model_constructor.zen_model.variables[
            "cost_capex_yearly_total"
        ] - model_constructor.zen_model.variables["cost_capex_yearly"].sum(
            ["set_technologies", "set_capacity_types", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        model_constructor.zen_model.add_constraint(
            "constraint_cost_capex_yearly_total", constraints
        )
