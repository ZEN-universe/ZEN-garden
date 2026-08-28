from zen_garden.model.component_types.constraint import GenericConstraint


class ChargeDischargeBinaryConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
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
        if not model_constructor.config.system.storage_charge_discharge_binary:
            return

        techs = model_constructor.zen_model.sets["set_storage_technologies"]
        nodes = model_constructor.zen_model.sets["set_nodes"]
        if len(techs) == 0:
            return
        # capacity limit as upper bound
        times = cls.get_storage2year_time_step_array(model_constructor)
        capacity_limit = model_constructor.zen_model.parameters.capacity_limit
        capacity_limit = cls.map_and_expand(capacity_limit, times)
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
            model_constructor.zen_model.variables["flow_storage_charge"]
            - model_constructor.zen_model.variables["charge_storage_binary"]
            * capacity_limit
        )
        rhs = 0
        constraint_charge = lhs <= rhs

        lhs = (
            model_constructor.zen_model.variables["flow_storage_discharge"]
            + model_constructor.zen_model.variables["charge_storage_binary"]
            * capacity_limit
        )
        rhs = capacity_limit
        constraint_discharge = lhs <= rhs

        model_constructor.zen_model.add_constraint(
            "constraint_charge_storage_binary", constraint_charge
        )
        model_constructor.zen_model.add_constraint(
            "constraint_discharge_storage_binary", constraint_discharge
        )
