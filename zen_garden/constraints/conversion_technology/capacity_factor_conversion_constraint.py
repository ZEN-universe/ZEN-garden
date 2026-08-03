import xarray as xr

from zen_garden.constraints.generic_constraint import GenericConstraint


class CapacityFactorConversionConstraint(GenericConstraint):
    def build(self):
        """Load is limited by the installed capacity and the maximum load factor.

        .. math::
            G_{i,n,t}^\\mathrm{r} \\leq m^{\\mathrm{max}}_{i,n,t}S_{i,n,y}

        :math:`m_{i,n,t}^{\\mathrm{max}}`: maximum load factor of the
        technology :math:`i` at node :math:`n` in time step :math:`t` \n
        :math:`S_{i,n,y}`: installed capacity of the technology :math:`i` at
        node :math:`n` in year :math:`y` \n
        :math:`G_{i,n,t}^\\mathrm{r}`: reference carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t`


        """
        techs = self.zen_model.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        times = self.zen_model.parameters.max_load.coords["set_time_steps_operation"]
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
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        )
        term_reference_flow = self.get_flow_expression_conversion(techs, nodes)
        lhs = term_capacity - term_reference_flow
        rhs = 0
        constraints = lhs >= rhs

        self.zen_model.add_constraint(
            "constraint_capacity_factor_conversion", constraints
        )
