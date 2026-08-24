import xarray as xr

from zen_garden.topology.generic_constraint import GenericConstraint


class CapacityFactorStorageConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Limits load of storage technologies by capacity and maximum load factor.

        Formulation:

        .. math::
            F^{\\mathrm{ch}}_{h,n,t}+F^{\\mathrm{dis}}_{h,n,t}\\leq
            \\ell^{\\mathrm{max}}_{h,n,t}K_{h,n,y}

        Notation:

        :math:`F^{\\mathrm{ch}}_{h,n,t}`: carrier flow into storage technology :math:`h`
        on node :math:`n` and time :math:`t` in year :math:`y`
        :math:`F^{\\mathrm{dis}}_{h,n,t}`: carrier flow out of storage
        technology :math:`h`on node :math:`n` and time :math:`t` in year :math:`y`
        :math:`\\ell^{\\mathrm{max}}_{h,n,t}`: maximum load factor for storage
        technology :math:`h` on node :math:`n` and time :math:`t` in year :math:`y`
        :math:`K_{h,n,y}`: storage capacity of storage technology :math:`h` on
        node :math:`n` in year :math:`y`
        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        times = self.zen_model.lp_model.variables.coords["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [self.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
        )
        term_capacity = (
            self.zen_model.parameters.max_load.loc[techs, nodes, :]
            * self.zen_model.variables["capacity"].loc[
                techs, "power", nodes, time_step_year
            ]
        ).rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )

        # TODO integrate level storage here as well
        lhs = term_capacity - self.get_flow_expression_storage(rename=False)
        rhs = 0
        constraints = lhs >= rhs
        ### return
        self.zen_model.add_constraint("constraint_capacity_factor_storage", constraints)
