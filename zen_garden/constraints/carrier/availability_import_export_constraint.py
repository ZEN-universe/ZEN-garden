from zen_garden.constraints.generic_constraint import GenericConstraint


class AvailabilityImportExportConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        node- and time-dependent carrier availability to import/export from outside
        the system boundaries.

        Formulation:

        .. math::
            F^{\\mathrm{imp}}_{c,n,t} \\leq a^{\\mathrm{imp}}_{c,n,t}

        .. math::
            F^{\\mathrm{exp}}_{c,n,t} \\leq a^{\\mathrm{exp}}_{c,n,t}

        Notation:

        :math:`F^{\\mathrm{imp}}_{c,n,t}`: flow of carrier :math:`c` imported
        at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{exp}}_{c,n,t}`: flow of carrier :math:`c` exported
        at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`a^{\\mathrm{imp}}_{c,n,t}`: availability of carrier :math:`c` to import
        at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`a^{\\mathrm{exp}}_{c,n,t}`: availability of carrier :math:`c` to export
        at node :math:`n` in time step :math:`t` of year :math:`y`
        """
        lhs_imp = self.zen_model.variables["flow_import"]
        rhs_imp = self.zen_model.parameters.availability_import
        constraints_imp = lhs_imp <= rhs_imp

        lhs_exp = self.zen_model.variables["flow_export"]
        rhs_exp = self.zen_model.parameters.availability_export
        constraints_exp = lhs_exp <= rhs_exp

        self.zen_model.add_constraint("constraint_availability_import", constraints_imp)
        self.zen_model.add_constraint("constraint_availability_export", constraints_exp)
