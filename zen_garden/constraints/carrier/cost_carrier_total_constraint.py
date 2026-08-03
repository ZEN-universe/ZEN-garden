from zen_garden.constraints.generic_constraint import GenericConstraint


class CostCarrierTotalConstraint(GenericConstraint):
    def build(self):
        """Total cost of importing and exporting carrier.

        .. math::
            C_y^{\\mathcal{C}} = \\sum_{c\\in\\mathcal{C}}\\sum_{n\\in\\mathcal{N}}
            \\sum_{t\\in\\mathcal{T}} \\tau_t (O_{c,n,t} + O_{c,n,t}^{\\mathrm{shed}\\
            \\mathrm{demand}})

        :math:`O_{c,n,t}`: cost of importing and exporting carrier :math:`c`
        at node :math:`n` and time step :math:`t`\n
        :math:`O_{c,n,t}^{\\mathrm{shed\\ demand}}`: cost of shedding demand
        of carrier :math:`c` at node :math:`n` and time step :math:`t`\n
        :math:`\\tau_t`: duration of time step :math:`t`


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
