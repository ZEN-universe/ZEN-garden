from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.storage_technology import StorageTechnology
from zen_garden.model.components.set_registry import SetRegistry
from zen_garden.utils import linexpr_from_tuple_np


class StorageTechnologyCapexConstraint(GenericConstraint):
    def build(self):
        """Definition of the capital expenditures for the storage technology.

        .. math::
            CAPEX_{y,n,i} = \\Delta S_{h,p,y} \\alpha_{k,n,y}

        :math:`\\Delta S_{h,p,y}`: capacity addition of storage technology :math:`h`
        on node :math:`n` in year :math:`y` \n
        :math:`\\alpha_{k,n,y}`: specific capex of storage technology :math:`k` on
        node :math:`n` in year :math:`y`
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
            return []

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
