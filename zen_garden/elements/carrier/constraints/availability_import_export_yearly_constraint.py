import numpy as np

from zen_garden.model.component_types.constraint import GenericConstraint


class AvailabilityImportExportYearlyConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        node- and year-dependent carrier availability to import/export from outside
        the system boundaries.

        Formulation:

        .. math::
            a^{\\mathrm{imp}}_{c,n,y} \\geq \\sum_{t\\in\\mathcal{T}_y}\\Delta t_t
            F^{\\mathrm{imp}}_{c,n,t}

        .. math::
            a^{\\mathrm{exp}}_{c,n,y} \\geq \\sum_{t\\in\\mathcal{T}_y}\\Delta t_t
            F^{\\mathrm{exp}}_{c,n,t}

        Notation:

        :math:`a^{\\mathrm{imp}}_{c,n,y}`: yearly availability of
        carrier :math:`c` to import at node :math:`n`
        :math:`a^{\\mathrm{exp}}_{c,n,y}`: yearly availability of
        carrier :math:`c` to export at node :math:`n`
        :math:`\\Delta t_t`: is the duration of time step :math:`t`
        :math:`F^{\\mathrm{imp}}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{exp}}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` in time step :math:`t` of year :math:`y`
        """
        # The constraint is only constrained if the availability is finite
        mask_imp = (
            model_constructor.zen_model.parameters.availability_import_yearly != np.inf
        )
        mask_exp = (
            model_constructor.zen_model.parameters.availability_export_yearly != np.inf
        )

        # import
        lhs_imp = (
            (
                model_constructor.zen_model.variables["flow_import"]
                * cls.get_year_time_step_duration_array(model_constructor)
            )
            .sum("set_time_steps_operation")
            .where(mask_imp)
        )
        rhs_imp = (
            model_constructor.zen_model.parameters.availability_import_yearly.where(
                mask_imp
            )
        )
        constraints_imp = lhs_imp <= rhs_imp

        # export
        lhs_exp = (
            (
                model_constructor.zen_model.variables["flow_export"]
                * cls.get_year_time_step_duration_array(model_constructor)
            )
            .sum("set_time_steps_operation")
            .where(mask_exp)
        )
        rhs_exp = (
            model_constructor.zen_model.parameters.availability_export_yearly.where(
                mask_exp
            )
        )
        constraints_exp = lhs_exp <= rhs_exp

        model_constructor.zen_model.add_constraint(
            "constraint_availability_import_yearly", constraints_imp
        )
        model_constructor.zen_model.add_constraint(
            "constraint_availability_export_yearly", constraints_exp
        )
