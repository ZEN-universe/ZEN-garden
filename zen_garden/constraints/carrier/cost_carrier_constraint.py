from zen_garden.constraints.generic_constraint import GenericConstraint


class CostCarrierConstraint(GenericConstraint):
    def build(self):
        """Cost of importing and exporting carrier.

        .. math::
        O_{c,n,t} = \\underline{u}_{c,n,t} \\underline{U}_{c,n,t} -
        \\overline{v}_{c,n,t} \\overline{U}_{c,n,t}

        :math:`\\underline{u}_{c,n,t}`: import price of carrier :math:`c` at
        node :math:`n` and time step :math:`t`\n
        :math:`\\overline{v}_{c,n,t}`: export price of carrier :math:`c` at
        node :math:`n` and time step :math:`t`\n
        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` and time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` and time step :math:`t`

        """
        ### formulate constraint
        lhs = (
            self.zen_model.variables["cost_carrier"]
            - self.zen_model.parameters.price_import
            * self.zen_model.variables["flow_import"]
            + self.zen_model.parameters.price_export
            * self.zen_model.variables["flow_export"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_carrier", constraints)
