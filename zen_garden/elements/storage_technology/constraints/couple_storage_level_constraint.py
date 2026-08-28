from zen_garden.model.component_types.constraint import GenericConstraint


class CoupleStorageLevelConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Couple subsequent storage levels (time coupling constraints).

        Formulation:

        .. math::
            \\begin{aligned}
            S^{\\mathrm{level}}_{h,n,\\tilde{t}}
            ={}&S^{\\mathrm{level}}_{h,n,\\tilde{t}-1}
            (1-\\lambda^{\\mathrm{self}}_{h,n})^{\\Delta \\tilde{t}_{\\tilde{t}}} \\\\
            &+\\left(\\eta^{\\mathrm{ch}}_{h,n,y}
            F^{\\mathrm{ch}}_{h,n,\\sigma(\\tilde{t})}
            -\\frac{F^{\\mathrm{dis}}_{h,n,\\sigma(\\tilde{t})}}
            {\\eta^{\\mathrm{dis}}_{h,n,y}}
            +q^{\\mathrm{in}}_{h,n,\\sigma(\\tilde{t})}
            -F^{\\mathrm{spill}}_{h,n,\\sigma(\\tilde{t})}\\right)
            \\sum_{\\tilde{t}'=0}^{\\Delta \\tilde{t}_{\\tilde{t}}-1}
            (1-\\lambda^{\\mathrm{self}}_{h,n})^{\\tilde{t}'}.
            \\end{aligned}

        Notation:

        :math:`S^{\\mathrm{level}}_{h,n,\\tilde{t}}`: storage level of storage
        technology :math:`h` at node :math:`n` in storage time step :math:`\\tilde{t}`
        of year :math:`y`
        :math:`\\lambda^{\\mathrm{self}}_{h,n}`: self-discharge rate of storage
        technology :math:`h` at node :math:`n`
        :math:`\\Delta \\tilde{t}_{\\tilde{t}}`: duration of storage time step of
        technology :math:`h`
        :math:`\\eta^{\\mathrm{ch}}_{h,n,y}`: efficiency during charging of storage
        technology :math:`h` at node :math:`n` in year :math:`y`
        :math:`\\eta^{\\mathrm{dis}}_{h,n,y}`: efficiency during discharging of storage
        technology :math:`h` at node :math:`n` in year :math:`y`
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
        techs = model_constructor.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        self_discharge = model_constructor.zen_model.parameters.self_discharge
        flow_storage_inflow = model_constructor.zen_model.parameters.flow_storage_inflow
        flow_storage_spillage = (
            model_constructor.zen_model.lp_model.variables.flow_storage_spillage
        )
        time_steps_storage_duration = (
            model_constructor.zen_model.parameters.time_steps_storage_duration
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
        times_coupling, mask_coupling = cls.get_previous_storage_time_step_array(
            model_constructor
        )
        self_discharge_previous = (1 - self_discharge) ** time_steps_storage_duration
        self_discharge_previous["set_time_steps_storage"] = times_coupling
        term_delta_storage_level = model_constructor.zen_model.variables[
            "storage_level"
        ] - self_discharge_previous * model_constructor.zen_model.variables[
            "storage_level"
        ].sel(
            {"set_time_steps_storage": times_coupling}
        )
        # charge and discharge flow
        times_year_time_step = cls.get_year_time_step_array(model_constructor)
        efficiency_charge = (
            model_constructor.zen_model.parameters.efficiency_charge.broadcast_like(
                times_year_time_step
            )
            .where(times_year_time_step, 0.0)
            .sum("set_years")
        )
        efficiency_discharge = (
            model_constructor.zen_model.parameters.efficiency_discharge.broadcast_like(
                times_year_time_step
            )
            .where(times_year_time_step, 0.0)
            .sum("set_years")
        )
        term_flow_charge_discharge = (
            model_constructor.zen_model.variables["flow_storage_charge"]
            * efficiency_charge
            - model_constructor.zen_model.variables[
                "flow_storage_discharge"
            ].to_linexpr()
            / efficiency_discharge
            + flow_storage_inflow
            - flow_storage_spillage
        )
        times_power2energy = cls.get_power2energy_time_step_array(model_constructor)
        term_flow_charge_discharge = cls.map_and_expand(
            term_flow_charge_discharge, times_power2energy
        )
        term_flow_charge_discharge = term_flow_charge_discharge * multiplier
        # sum up all terms
        lhs = (term_delta_storage_level - term_flow_charge_discharge).where(
            mask_coupling, 0.0
        )
        rhs = 0
        constraints = lhs == rhs

        model_constructor.zen_model.add_constraint(
            "constraint_couple_storage_level", constraints
        )
