import xarray as xr

from zen_garden.constraints.generic_constraint import GenericConstraint


class CapacityFactorTransportConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        Load is limited by the installed capacity and the maximum load factor.

        Formulation:

        .. math::
            F^{\mathrm{trans}}_{h,e,t} \\leq \\ell^{\\mathrm{max}}_{h,e,t}K_{h,e,y}

        Notation:

        :math:`F^{\mathrm{trans}}_{h,e,t}`: carrier flow through transport 
        technology :math:`h`
        on edge :math:`e` in time step :math:`t` of year :math:`y`
        :math:`\\ell^{\\mathrm{max}}_{h,e,t}`: Maximum load factor of transport
        technology :math:`h` on edge :math:`e` in time step :math:`t` of year
        :math:`y`
        :math:`K_{h,e,y}`: Capacity of transport technology :math:`h` on
        edge :math:`e` in year :math:`y`
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
