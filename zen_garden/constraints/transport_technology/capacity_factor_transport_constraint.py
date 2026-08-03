import xarray as xr

from zen_garden.constraints.generic_constraint import GenericConstraint


class CapacityFactorTransportConstraint(GenericConstraint):
    def build(self):
        """Load is limited by the installed capacity and the maximum load factor.

        .. math::
            F_{j,e,t,y}^\\mathrm{r} \\leq m^{\\mathrm{max}}_{j,e,t,y}S_{j,e,y}


        :math:`F_{j,e,t,y}^\\mathrm{r}`: Reference flow of carrier through transport
        technology :math:`j` on edge :math:`i` and time :math:`t` in year :math:`y` \n
        :math:`m^{\\mathrm{max}}_{j,e,t,y}`: Maximum load factor of transport
        technology :math:`j` on edge :math:`i` and time :math:`t` in year :math:`y` \n
        :math:`S_{j,e,y}`: Capacity of transport technology :math:`j` on
        edge :math:`i` in year :math:`y`


        """
        techs = self.zen_model.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = self.zen_model.sets["set_edges"]
        times = self.zen_model.variables["flow_transport"].coords[
            "set_time_steps_operation"
        ]
        time_step_year = xr.DataArray(
            [self.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
        )
        term_capacity = (
            self.zen_model.parameters.max_load.loc[techs, edges, :]
            * self.zen_model.variables["capacity"].loc[
                techs, "power", edges, time_step_year
            ]
        ).rename(
            {
                "set_technologies": "set_transport_technologies",
                "set_location": "set_edges",
            }
        )

        lhs = (
            term_capacity
            - self.zen_model.variables["flow_transport"].loc[techs, edges, :]
        )
        rhs = 0
        constraints = lhs >= rhs
        ### return
        self.zen_model.add_constraint(
            "constraint_capacity_factor_transport", constraints
        )
