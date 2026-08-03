import pandas as pd

from zen_garden.constraints.generic_constraint import GenericConstraint


class CostOpexYearlyConstraint(GenericConstraint):
    def build(self):
        """Yearly opex for a technology at a location in each year.

        .. math::
            OPEX_{h,p,y} = \\sum_{t\\in\\mathcal{T}}\\tau_t O_{h,p,t}^t
            + \\gamma_{h,y} S_{h,p,y} + \\gamma_{k,y}^\\mathrm{e} S_{k,n,y}^\\mathrm{e}

        :math:`OPEX_{h,p,y}`: opex of operating technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\tau_t`: duration of time step :math:`t` \n
        :math:`O_{h,p,t}^t`: variable opex of operating technology :math:`h` at
        location :math:`p` in time step :math:`t` \n
        :math:`\\gamma_{h,y}`: specific fixed opex of technology :math:`h` in
        year :math:`y` \n
        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\gamma_{k,y}^\\mathrm{e}`: specific fixed opex of storage
        technology :math:`k` in year :math:`y` \n
        :math:`S_{k,n,y}^\\mathrm{e}`: installed capacity of storage
        technology :math:`k` at node :math:`n` in year :math:`y`

        """
        times_dict: dict[str, pd.Series] = {
            y: self.zen_model.parameters.time_steps_operation_duration.loc[
                self.time_steps.get_time_steps_year2operation(y)
            ].to_series()
            for y in self.zen_model.sets["set_years"]
        }
        times = pd.concat(times_dict, keys=times_dict.keys())
        times.index.names = ["set_years", "set_time_steps_operation"]
        times = times.to_xarray().broadcast_like(
            self.zen_model.variables["cost_opex_variable"].mask
        )
        term_opex_variable = (
            self.zen_model.variables["cost_opex_variable"] * times
        ).sum("set_time_steps_operation")
        term_opex_fixed = (
            self.zen_model.parameters.opex_specific_fixed
            * self.zen_model.variables["capacity"]
        ).sum("set_capacity_types")
        lhs = (
            self.zen_model.variables["cost_opex_yearly"]
            - term_opex_variable
            - term_opex_fixed
        )
        rhs = 0
        constraints = lhs == rhs

        ### return
        self.zen_model.add_constraint("constraint_cost_opex_yearly", constraints)
