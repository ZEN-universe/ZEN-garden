from zen_garden.topology.generic_constraint import GenericConstraint


class StorageLevelMaxConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Limit maximum storage level to capacity.

        Formulation:

        .. math::
            S^{\\mathrm{level}}_{h,n,\\tilde{t}} \\le K^{\\mathrm{energy}}_{h,n,y}

        Notation:

        :math:`S^{\\mathrm{level}}_{h,n,\\tilde{t}}`: storage level of storage
        technology :math:`h` at node :math:`n` in storage time step :math:`\\tilde{t}`
        of year :math:`y`
        :math:`K^{\\mathrm{energy}}_{h,n,y}`: energy capacity of storage technology
        :math:`h` on node :math:`n` in year :math:`y`
        """
        techs = self.zen_model.sets["set_storage_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        if len(techs) == 0:
            return
        # mask for energy capacity and storage time steps
        times = self.get_storage2year_time_step_array()
        capacity = self.map_and_expand(self.zen_model.variables["capacity"], times)
        capacity = capacity.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        capacity = capacity.sel({"set_nodes": nodes, "set_storage_technologies": techs})
        storage_level = self.zen_model.variables["storage_level"]
        mask_capacity_type = (
            self.zen_model.variables["capacity"].coords["set_capacity_types"]
            == "energy"
        )
        lhs = (storage_level - capacity).where(mask_capacity_type, 0.0)
        rhs = 0
        constraints = lhs <= rhs

        self.zen_model.add_constraint("constraint_storage_level_max", constraints)
