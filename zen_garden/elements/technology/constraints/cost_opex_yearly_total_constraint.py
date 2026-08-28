from zen_garden.topology.generic_constraint import GenericConstraint


class CostOpexYearlyTotalConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Sums over all technologies to calculate total opex.

        Formulation:

        .. math::
            C^{\\mathrm{op}}_y = \\sum_{h\\in\\mathcal{H}}
            \\sum_{p\\in\\mathcal{P}_h} C^{\\mathrm{op,ann}}_{h,p,y}

        Notation:

        :math:`C^{\\mathrm{op,ann}}_{h,p,y}`: OPEX of operating technology :math:`h` at
        location :math:`p` in year :math:`y`
        """
        lhs = model_constructor.zen_model.variables[
            "cost_opex_yearly_total"
        ] - model_constructor.zen_model.variables["cost_opex_yearly"].sum(
            ["set_technologies", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        model_constructor.zen_model.add_constraint(
            "constraint_cost_opex_yearly_total", constraints
        )
