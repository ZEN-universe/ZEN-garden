from zen_garden.constraints.generic_constraint import GenericConstraint


class FlowStorageSpillageConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Limit storage spillage to the exogenous storage inflow.

        Formulation:

        .. math::
            0 \\leq F^{\\mathrm{spill}}_{h,n,t} \\leq q^{\\mathrm{in}}_{h,n,t}

        The lower bound is imposed by the nonnegative variable definition.

        Notation:

        :math:`F^{\\mathrm{spill}}_{h,n,t}`: storage spillage of storage technology
        :math:`h`
        at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`q^{\\mathrm{in}}_{h,n,t}`: exogenous inflow into storage technology
        :math:`h` at node :math:`n` in time step :math:`t` of year :math:`y`
        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return

        flow_storage_inflow = self.zen_model.parameters.flow_storage_inflow
        flow_storage_spillage = self.zen_model.lp_model.variables.flow_storage_spillage

        lhs = flow_storage_spillage - flow_storage_inflow
        rhs = 0
        constraints = lhs <= rhs

        self.zen_model.add_constraint("constraint_flow_storage_spillage", constraints)
