import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class CapacityEnergyToPowerRatioConstraint(GenericConstraint):
    def build(self):
        """Limit capacity power to energy ratio.

        .. math::
            \\rho_k^{min} S^{e}_{k,n,y} \\le S_{k,n,y}

        .. math::
            S_{k,n,y} \\le \\rho_k^{max} S^{e}_{k,n,y}

        :math:`S^{\\mathrm{power}}_{k,n,y}`: installed capacity in terms of power of
        storage :math:`k` at node :math:`n` in year :math:`y` \n
        :math:`S^{e}_{k,n,y}`: installed capacity in terms of energy of
        storage :math:`k` at node :math:`n` in year :math:`y` \n
        :math:`\\rho_k^{min}`: minimum power-to-energy ratio of storage :math:`k` \n
        :math:`\\rho_k^{max}`: maximum power-to-energy ratio of storage :math:`k`

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return None
        e2p_min = self.zen_model.parameters.energy_to_power_ratio_min
        e2p_max = self.zen_model.parameters.energy_to_power_ratio_max
        mask_min = e2p_min != np.inf
        mask_max = e2p_max != np.inf

        capacity_addition = self.zen_model.variables["capacity_addition"].rename(
            {"set_technologies": "set_storage_technologies"}
        )
        capacity_addition_power = capacity_addition.sel(
            {"set_storage_technologies": techs, "set_capacity_types": "power"}
        )
        capacity_addition_energy = capacity_addition.sel(
            {"set_storage_technologies": techs, "set_capacity_types": "energy"}
        )
        lhs = (capacity_addition_energy - capacity_addition_power * e2p_min).where(
            mask_min
        )
        rhs = 0
        constraints_min = lhs >= rhs
        lhs = (capacity_addition_energy - capacity_addition_power * e2p_max).where(
            mask_max
        )
        constraints_max = lhs <= rhs

        self.zen_model.add_constraint(
            "constraint_capacity_energy_to_power_ratio_min", constraints_min
        )
        self.zen_model.add_constraint(
            "constraint_capacity_energy_to_power_ratio_max", constraints_max
        )
