from zen_garden.constraints.generic_constraint import GenericConstraint


class CoupleStorageLevelConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        Couple subsequent storage levels (time coupling constraints).

        Formulation:

        .. math::
            S^{\\mathrm{level}}_{h,n,\\tilde{t}} = 
            S^{\\mathrm{level}}_{h,n,\\tilde{t}-1,y}
            (1-\\lambda^{\\mathrm{self}}_h)^{\\Delta \\tilde{t}_{\\tilde{t}}} +
            (\\eta^{\\mathrm{ch}}_h F^{\\mathrm{ch}}_{h,n,\\sigma(\\tilde{t})} -
            \\frac{F^{\\mathrm{dis}}_{h,n,\\sigma(\\tilde{t})}}
            {\\eta^{\\mathrm{dis}}_h}
            + q^{\\mathrm{in}}_{h,n,\\sigma(\\tilde{t})}
            - F^{\\mathrm{spill}}_{h,n,\\sigma(\\tilde{t})})
            \\sum^{\\Delta \\tilde{t}_{\\tilde{t}}-1}_{\\tilde{t}'=0}
            (1-\\lambda^{\\mathrm{self}}_h)^{\\tilde{t}'}

        Notation:

        :math:`S^{\\mathrm{level}}_{h,n,\\tilde{t}}`: storage level of storage technology
        :math:`h` at node :math:`n` in storage time step :math:`\\tilde{t}`
        of year :math:`y`
        :math:`\\lambda^{\\mathrm{self}}_h`: self-discharge rate of storage 
        technology :math:`h`
        :math:`\\Delta \\tilde{t}_{\\tilde{t}}`: duration of storage time step of
        technology :math:`h`
        :math:`\\eta^{\\mathrm{ch}}_h`: efficiency during charging of storage
        technology :math:`h`
        :math:`\\eta^{\\mathrm{dis}}_h`: efficiency during discharging of storage
        technology :math:`h`
        :math:`F^{\\mathrm{ch}}_{h,n,\\sigma(\\tilde{t})}`: charge flow into storage
        technology :math:`h` at node :math:`n` and time
        :math:`\\sigma(\\tilde{t})` in year :math:`y`
        :math:`F^{\\mathrm{dis}}_{h,n,\\sigma(\\tilde{t})}`: discharge flow out of
        storage technology :math:`h` at node :math:`n` and time
        :math:`\\sigma(\\tilde{t})` in
        year :math:`y`
        :math:`q^{\\mathrm{in}}_{h,n,\\sigma(\\tilde{t})}`: exogenous inflow into 
        storage
        :math:`F^{\\mathrm{spill}}_{h,n,\\sigma(\\tilde{t})}`: storage spillage
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
