from zen_garden.constraints.generic_constraint import GenericConstraint


class CoupleStorageLevelConstraint(GenericConstraint):
    def build(self):
        """Couple subsequent storage levels (time coupling constraints).

        .. math::
            L_{k,n,t^k,y} = L_{k,n,t^k-1,y} (1-\\phi_k)^{\\tau_{t^k}^k} +
            (\\underline{\\eta}_k \\underline{H}_{k,n,\\sigma(t^k),y} -
            \\frac{\\overline{H}_{k,n,\\sigma(t^k),y}}{\\overline{\\eta}_k})
            \\sum^{\\tau_{t^k}^k-1}_{\\tilde{t}^k=0} (1-\\phi_k)^{\\tilde{t}^k}

        :math:`L_{k,n,t^k,y}`: storage level of storage technology :math:`k` on
        node :math:`n` and time :math:`t^k` in year :math:`y` \n
        :math:`\\phi_k`: self discharge rate of storage technology :math:`k` \n
        :math:`\\tau_{t^k}^k`: duration of storage level time step of storage
        technology :math:`k` \n
        :math:`\\underline{\\eta}_k`: efficiency during charging of storage
        technology :math:`k` \n
        :math:`\\overline{\\eta}_k`: efficiency during discharging of storage
        technology :math:`k` \n
        :math:`\\underline{H}_{k,n,\\sigma(t^k),y}`: charge flow into storage
        technology :math:`k` on node :math:`n` and time :math:`\\sigma(t^k)` in
        year :math:`y` \n
        :math:`\\overline{H}_{k,n,\\sigma(t^k),y}`: discharge flow out of storage
        technology :math:`k` on node :math:`n` and time :math:`\\sigma(t^k)` in
        year :math:`y`

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        self_discharge = self.zen_model.parameters.self_discharge
        flow_storage_inflow = self.zen_model.parameters.flow_storage_inflow
        flow_storage_spillage = self.zen_model.lp_model.variables.flow_storage_spillage
        time_steps_storage_duration = (
            self.zen_model.parameters.time_steps_storage_duration
        )
        # reformulate self discharge multiplier as partial geometric series
        multiplier_w_discharge = (
            1 - (1 - self_discharge) ** time_steps_storage_duration
        ) / (1 - (1 - self_discharge))
        multiplier_wo_discharge = time_steps_storage_duration
        multiplier = multiplier_w_discharge.where(
            self_discharge != 0, 0.0
        ) + multiplier_wo_discharge.where(self_discharge == 0, 0.0)
        # time coupling to previous time step
        times_coupling, mask_coupling = self.get_previous_storage_time_step_array()
        self_discharge_previous = (1 - self_discharge) ** time_steps_storage_duration
        self_discharge_previous["set_time_steps_storage"] = times_coupling
        term_delta_storage_level = self.zen_model.variables[
            "storage_level"
        ] - self_discharge_previous * self.zen_model.variables["storage_level"].sel(
            {"set_time_steps_storage": times_coupling}
        )
        # charge and discharge flow
        times_year_time_step = self.get_year_time_step_array()
        efficiency_charge = (
            self.zen_model.parameters.efficiency_charge.broadcast_like(
                times_year_time_step
            )
            .where(times_year_time_step, 0.0)
            .sum("set_years")
        )
        efficiency_discharge = (
            self.zen_model.parameters.efficiency_discharge.broadcast_like(
                times_year_time_step
            )
            .where(times_year_time_step, 0.0)
            .sum("set_years")
        )
        term_flow_charge_discharge = (
            self.zen_model.variables["flow_storage_charge"] * efficiency_charge
            - self.zen_model.variables["flow_storage_discharge"].to_linexpr()
            / efficiency_discharge
            + flow_storage_inflow
            - flow_storage_spillage
        )
        times_power2energy = self.get_power2energy_time_step_array()
        term_flow_charge_discharge = self.map_and_expand(
            term_flow_charge_discharge, times_power2energy
        )
        term_flow_charge_discharge = term_flow_charge_discharge * multiplier
        # sum up all terms
        lhs = (term_delta_storage_level - term_flow_charge_discharge).where(
            mask_coupling, 0.0
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_couple_storage_level", constraints)
