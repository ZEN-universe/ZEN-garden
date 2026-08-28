import pandas as pd

from zen_garden.model.component_types.constraint import GenericConstraint


class CostOpexYearlyConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Yearly opex for a technology at a location in each year.

        Formulation:

        .. math::
            C^{\\mathrm{op,ann}}_{h,p,y} =
            \\sum_{t\\in\\mathcal{T}_y}\\Delta t_t C^{\\mathrm{op,var}}_{h,p,t}
            + \\kappa^{\\mathrm{op,fix}}_{h,p,y} K_{h,p,y}

        For storage technologies, the implementation additionally includes the
        corresponding energy-capacity term:

        .. math::
            \\kappa^{\\mathrm{op,fix,energy}}_{h,n,y}K^{\\mathrm{energy}}_{h,n,y}

        Notation:

        :math:`C^{\\mathrm{op,ann}}_{h,p,y}`: OPEX of operating technology :math:`h` at
        location :math:`p` in year :math:`y`
        :math:`\\Delta t_t`: duration of time step :math:`t`
        :math:`C^{\\mathrm{op,var}}_{h,p,t}`: variable OPEX of operating technology
        :math:`h` at location :math:`p` in time step :math:`t` of year :math:`y`
        :math:`\\kappa^{\\mathrm{op,fix}}_{h,p,y}`: specific fixed OPEX
        :math:`K_{h,p,y}`: installed capacity of technology :math:`h` at location
        :math:`p` in year :math:`y`
        """
        optimization_model = model_constructor.optimization_model
        times_dict: dict[str, pd.Series] = {
            y: optimization_model.parameters.time_steps_operation_duration.loc[
                model_constructor.time_steps.get_time_steps_year2operation(y)
            ].to_series()
            for y in optimization_model.sets["set_years"]
        }
        times = pd.concat(times_dict, keys=times_dict.keys())
        times.index.names = ["set_years", "set_time_steps_operation"]
        times = times.to_xarray().broadcast_like(
            optimization_model.variables["cost_opex_variable"].mask
        )
        term_opex_variable = (
            optimization_model.variables["cost_opex_variable"] * times
        ).sum("set_time_steps_operation")
        term_opex_fixed = (
            optimization_model.parameters.opex_specific_fixed
            * optimization_model.variables["capacity"]
        ).sum("set_capacity_types")
        lhs = (
            optimization_model.variables["cost_opex_yearly"]
            - term_opex_variable
            - term_opex_fixed
        )
        rhs = 0
        constraints = lhs == rhs

        ### return
        optimization_model.add_constraint("constraint_cost_opex_yearly", constraints)
