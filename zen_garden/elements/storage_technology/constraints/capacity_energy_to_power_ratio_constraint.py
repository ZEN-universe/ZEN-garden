import numpy as np

from zen_garden.model.component_types.constraint import GenericConstraint


class CapacityEnergyToPowerRatioConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Limit the energy-to-power ratio of capacity additions.

        Formulation:

        .. math::
            r^{\\mathrm{EP,min}}_h \\Delta K_{h,n,y}
            \\leq \\Delta K^{\\mathrm{e}}_{h,n,y}

        .. math::
            \\Delta K^{\\mathrm{e}}_{h,n,y}
            \\leq r^{\\mathrm{EP,max}}_h \\Delta K_{h,n,y}

        Notation:

        :math:`\\Delta K_{h,n,y}`: power-capacity addition of
        storage :math:`h` at node :math:`n` in year :math:`y`
        :math:`\\Delta K^{\\mathrm{e}}_{h,n,y}`: energy-capacity addition of
        storage :math:`h` at node :math:`n` in year :math:`y`
        :math:`r^{\\mathrm{EP,min}}_h`: minimum energy-to-power ratio of storage
        :math:`h`
        :math:`r^{\\mathrm{EP,max}}_h`: maximum energy-to-power ratio of storage
        :math:`h`
        """
        techs = model_constructor.optimization_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return None
        e2p_min = (
            model_constructor.optimization_model.parameters.energy_to_power_ratio_min
        )
        e2p_max = (
            model_constructor.optimization_model.parameters.energy_to_power_ratio_max
        )
        mask_min = e2p_min != np.inf
        mask_max = e2p_max != np.inf

        capacity_addition = model_constructor.optimization_model.variables[
            "capacity_addition"
        ].rename({"set_technologies": "set_storage_technologies"})
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

        model_constructor.optimization_model.add_constraint(
            "constraint_capacity_energy_to_power_ratio_min", constraints_min
        )
        model_constructor.optimization_model.add_constraint(
            "constraint_capacity_energy_to_power_ratio_max", constraints_max
        )
