from zen_garden.constraints.generic_constraint import GenericConstraint


class AvailabilityImportExportConstraint(GenericConstraint):
    def build(self):
        """node- and time-dependent carrier availability to import/export from outside
        the system boundaries.

        .. math::
            \\underline{U}_{c,n,t} \\leq \\underline{a}_{c,n,t}

        .. math::
            \\overline{U}_{c,n,t} \\leq \\overline{a}_{c,n,t}

        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported
        at node :math:`n` and time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported
        at node :math:`n` and time step :math:`t`\n
        :math:`\\underline{a}_{c,n,t}`: availability of carrier :math:`c` to import
        at node :math:`n` and time step :math:`t`\n
        :math:`\\overline{a}_{c,n,t}`: availability of carrier :math:`c` to export
        at node :math:`n` and time step :math:`t`

        """
        lhs_imp = self.zen_model.variables["flow_import"]
        rhs_imp = self.zen_model.parameters.availability_import
        constraints_imp = lhs_imp <= rhs_imp

        lhs_exp = self.zen_model.variables["flow_export"]
        rhs_exp = self.zen_model.parameters.availability_export
        constraints_exp = lhs_exp <= rhs_exp

        self.zen_model.add_constraint("constraint_availability_import", constraints_imp)
        self.zen_model.add_constraint("constraint_availability_export", constraints_exp)
