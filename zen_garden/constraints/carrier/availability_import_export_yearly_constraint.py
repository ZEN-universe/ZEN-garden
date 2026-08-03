import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class AvailabilityImportExportYearlyConstraint(GenericConstraint):
    def build(self):
        """node- and year-dependent carrier availability to import/export from outside
        the system boundaries.

        .. math::
            \\underline{a}_{c,n,y}^\\mathrm{Y} \\geq \\sum_{t\\in\\mathcal{T}}\\tau_t
            \\underline{U}_{c,n,t}

        .. math::
            \\overline{a}_{c,n,y}^\\mathrm{Y} \\geq \\sum_{t\\in\\mathcal{T}}\\tau_t
            \\overline{U}_{c,n,t}

        :math:`\\underline{a}_{c,n,y}^\\mathrm{Y}`: yearly availability of
        carrier :math:`c` to import at node :math:`n`\n
        :math:`\\overline{a}_{c,n,y}^\\mathrm{Y}`: yearly availability of
        carrier :math:`c` to export at node :math:`n`\n
        :math:`\\tau_t`: is the duration of time step :math:`t`\n
        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` at time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` at time step :math:`t`


        """
        # The constraint is only constrained if the availability is finite
        mask_imp = self.zen_model.parameters.availability_import_yearly != np.inf
        mask_exp = self.zen_model.parameters.availability_export_yearly != np.inf

        # import
        lhs_imp = (
            (
                self.zen_model.variables["flow_import"]
                * self.get_year_time_step_duration_array()
            )
            .sum("set_time_steps_operation")
            .where(mask_imp)
        )
        rhs_imp = self.zen_model.parameters.availability_import_yearly.where(mask_imp)
        constraints_imp = lhs_imp <= rhs_imp

        # export
        lhs_exp = (
            (
                self.zen_model.variables["flow_export"]
                * self.get_year_time_step_duration_array()
            )
            .sum("set_time_steps_operation")
            .where(mask_exp)
        )
        rhs_exp = self.zen_model.parameters.availability_export_yearly.where(mask_exp)
        constraints_exp = lhs_exp <= rhs_exp

        self.zen_model.add_constraint(
            "constraint_availability_import_yearly", constraints_imp
        )
        self.zen_model.add_constraint(
            "constraint_availability_export_yearly", constraints_exp
        )
