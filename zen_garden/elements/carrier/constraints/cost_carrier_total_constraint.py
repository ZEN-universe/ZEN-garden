from zen_garden.topology.generic_constraint import GenericConstraint


class CostCarrierTotalConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Total cost of importing and exporting carrier.

        Formulation:

        .. math::
            C^{\\mathrm{carrier}}_y = \\sum_{c\\in\\mathcal{C}}\\sum_{n\\in\\mathcal{N}}
            \\sum_{t\\in\\mathcal{T}_y} \\Delta t_t (C^{\\mathrm{carrier}}_{c,n,t}
            + C^{\\mathrm{shed}}_{c,n,t})

        Notation:

        :math:`C^{\\mathrm{carrier}}_{c,n,t}`: cost of importing and exporting carrier
        :math:`c` at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`C^{\\mathrm{shed}}_{c,n,t}`: cost of shedding
        demand of carrier :math:`c` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        :math:`\\Delta t_t`: duration of time step :math:`t`
        """
        times = self.get_year_time_step_duration_array()
        term_summed_cost_carrier = (
            (
                self.zen_model.variables["cost_carrier"].broadcast_like(times)
                + self.zen_model.variables["cost_shed_demand"].broadcast_like(times)
            )
            * times
        ).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])
        lhs = self.zen_model.variables["cost_carrier_total"] - term_summed_cost_carrier
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_carrier_total", constraints)
