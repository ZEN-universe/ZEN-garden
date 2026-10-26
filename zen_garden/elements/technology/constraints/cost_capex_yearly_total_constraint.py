from zen_garden.elements.storage_technology.variables.cost_capex_yearly_energy import (
    CostCapexYearlyEnergy,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.variables.cost_capex_yearly import CostCapexYearly
from zen_garden.elements.technology.variables.cost_capex_yearly_total import (
    CostCapexYearlyTotal,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CostCapexYearlyTotalConstraint(GenericConstraint):
    """Cost Capex Yearly Total."""

    name = "constraint_cost_capex_yearly_total"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Sums over all technologies to calculate total capex.

        Formulation:

        .. math::
            C^{\\mathrm{cap}}_y = \\sum_{h\\in\\mathcal{H}}
            \\sum_{p\\in\\mathcal{P}_h}C^{\\mathrm{cap,ann}}_{h,p,y}

        Notation:

        :math:`C^{\\mathrm{cap,ann}}_{h,p,y}`: annual CAPEX of technology :math:`h` at
        location :math:`p` in year :math:`y`
        :math:`p` in year :math:`y`, including all applicable capacity types

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        power_cost = model.variables[CostCapexYearly].sum(
            ["set_technologies", "set_location"]
        )
        if not model.sets[SetStorageTechnologies]:
            return model.variables[CostCapexYearlyTotal] - power_cost == 0
        energy_cost = model.variables[CostCapexYearlyEnergy].sum(
            ["set_storage_technologies", "set_nodes"]
        )
        lhs = model.variables[CostCapexYearlyTotal] - power_cost - energy_cost
        rhs = 0
        constraints = lhs == rhs

        return constraints
