from zen_garden.constraints.generic_constraint import GenericConstraint


class FlowStorageSpillageConstraint(GenericConstraint):
    def build(self):
        """Ensure that flow_energy_spillage is not greater than the flow_storage_inflow.

        .. math::

        Todo:

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
