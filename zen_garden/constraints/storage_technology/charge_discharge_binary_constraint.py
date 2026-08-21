from zen_garden.constraints.generic_constraint import GenericConstraint


class ChargeDischargeBinaryConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        Avoid simultaneous charge and discharge of storage technologies.

        Ensure that the storage technology cannot charge and discharge simultaneously
        within the same operational time step. This is only active if the
        storage_charge_discharge_binary flag in the system.json file is set to True.

        Formulation:

        .. math::
            F^{\\mathrm{ch}}_{h,n,t} \\le z^{\\mathrm{ch}}_{h,n,t}
            s^\\mathrm{max,power}_{h,n,y}

        .. math::
            F^{\\mathrm{dis}}_{h,n,t} \\le (1-z^{\\mathrm{ch}}_{h,n,t})
            s^\\mathrm{max,power}_{h,n,y}

        Notation:

        :math:`F^{\\mathrm{ch}}_{h,n,t}`: carrier flow into storage technology :math:`h`
        on node :math:`n` and time :math:`t` in year :math:`y`
        :math:`F^{\\mathrm{dis}}_{h,n,t}`: carrier flow out of storage
        technology :math:`h`on node :math:`n` and time :math:`t` in year :math:`y`
        :math:`s^\\mathrm{max,power}_{h,n,y}`: power-capacity limit of storage
        technology :math:`h` at location :math:`n` in year :math:`y`
        :math:`z^{\\mathrm{ch}}_{h,n,t}`: binary variable indicating whether the
        storage technology :math:`h` is in charging mode (1) or discharging mode (0) at
        location :math:`n` at time step :math:`t` in year :math:`y`
        """
        if not self.config.system.storage_charge_discharge_binary:
            return

        techs = self.zen_model.sets["set_storage_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        if len(techs) == 0:
            return
        # capacity limit as upper bound
        times = self.get_storage2year_time_step_array()
        capacity_limit = self.zen_model.parameters.capacity_limit
        capacity_limit = self.map_and_expand(capacity_limit, times)
        capacity_limit = capacity_limit.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        capacity_limit = capacity_limit.sel(
            {
                "set_nodes": nodes,
                "set_storage_technologies": techs,
                "set_capacity_types": "power",
            }
        )
        capacity_limit = capacity_limit.rename(
            {"set_time_steps_storage": "set_time_steps_operation"}
        )

        lhs = (
            self.zen_model.variables["flow_storage_charge"]
            - self.zen_model.variables["charge_storage_binary"] * capacity_limit
        )
        rhs = 0
        constraint_charge = lhs <= rhs

        lhs = (
            self.zen_model.variables["flow_storage_discharge"]
            + self.zen_model.variables["charge_storage_binary"] * capacity_limit
        )
        rhs = capacity_limit
        constraint_discharge = lhs <= rhs

        self.zen_model.add_constraint(
            "constraint_charge_storage_binary", constraint_charge
        )
        self.zen_model.add_constraint(
            "constraint_discharge_storage_binary", constraint_discharge
        )
