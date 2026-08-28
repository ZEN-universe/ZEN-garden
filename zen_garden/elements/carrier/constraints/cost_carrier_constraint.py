from zen_garden.topology.generic_constraint import GenericConstraint


class CostCarrierConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
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
            model_constructor.zen_model.variables["cost_carrier"]
            - model_constructor.zen_model.parameters.price_import
            * model_constructor.zen_model.variables["flow_import"]
            + model_constructor.zen_model.parameters.price_export
            * model_constructor.zen_model.variables["flow_export"]
        )
        rhs = 0
        constraints = lhs == rhs

        model_constructor.zen_model.add_constraint(
            "constraint_cost_carrier", constraints
        )
