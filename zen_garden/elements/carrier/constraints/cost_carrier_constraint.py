from zen_garden.constraints.generic_constraint import GenericConstraint


class CostCarrierConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Cost of importing and exporting carrier.

        Formulation:

        .. math::
            C^{\\mathrm{carrier}}_{c,n,t} = \\pi^{\\mathrm{imp}}_{c,n,t}
            F^{\\mathrm{imp}}_{c,n,t} - \\pi^{\\mathrm{exp}}_{c,n,t}
            F^{\\mathrm{exp}}_{c,n,t}

        Notation:

        :math:`\\pi^{\\mathrm{imp}}_{c,n,t}`: import price of carrier :math:`c` at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`\\pi^{\\mathrm{exp}}_{c,n,t}`: export price of carrier :math:`c` at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{imp}}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{exp}}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` in time step :math:`t` of year :math:`y`
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
