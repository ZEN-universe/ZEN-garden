from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.storage_technology import StorageTechnology
from zen_garden.model.components.set_registry import SetRegistry
from zen_garden.utils import linexpr_from_tuple_np


class StorageTechnologyCapexConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        Definition of the capital expenditures for the storage technology.

        Formulation:

        .. math::
            C^{\\mathrm{cap,overnight}}_{h,n,y} = 
            \\kappa^{\\mathrm{cap,power}}_{h,y}\\Delta K_{h,n,y}
            + \\kappa^{\\mathrm{cap,energy}}_{h,y}\\Delta K^{\\mathrm{energy}}_{h,n,y}

        The implementation stores the two terms separately along its capacity-type
        dimension; their sum is :math:`C^{\\mathrm{cap,overnight}}_{h,n,y}`.

        Notation:

        :math:`C^{\\mathrm{cap,overnight}}_{h,n,y}`: total overnight CAPEX of storage 
        technology :math:`h` at node :math:`n` in year :math:`y`
        :math:`\\Delta K_{h,n,y}` and :math:`\\Delta K^{\\mathrm{energy}}_{h,n,y}`:
        power- and energy-capacity additions
        :math:`\\kappa^{\\mathrm{cap,power}}_{h,y}` and :math:
        `\\kappa^{\\mathrm{cap,energy}}_{h,y}`: specific power- and energy-capacity 
        CAPEX
        """
        index_values, index_names = self.zen_model.create_custom_set(
            [
                "set_storage_technologies",
                "set_capacity_types",
                "set_nodes",
                "set_years",
            ],
            StorageTechnology,
        )

        # check if we need to continue
        if len(index_values) == 0:
            return

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # get all the arrays and coords
        techs, capacity_types, nodes, times = SetRegistry.tuple_to_arr(
            index_values, index_names, unique=True
        )
        coords = [
            self.zen_model.lp_model.variables.coords["set_storage_technologies"],
            self.zen_model.lp_model.variables.coords["set_capacity_types"],
            self.zen_model.lp_model.variables.coords["set_nodes"],
            self.zen_model.lp_model.variables.coords["set_years"],
        ]

        ### formulate constraint
        lhs = linexpr_from_tuple_np(
            [
                (
                    1.0,
                    self.zen_model.variables["cost_capex_overnight"].loc[
                        techs, capacity_types, nodes, times
                    ],
                ),
                (
                    -self.zen_model.parameters.capex_specific_storage.loc[
                        techs, capacity_types, nodes, times
                    ],
                    self.zen_model.variables["capacity_addition"].loc[
                        techs, capacity_types, nodes, times
                    ],
                ),
            ],
            coords,
            self.zen_model.lp_model,
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_storage_technology_capex", constraints
        )
