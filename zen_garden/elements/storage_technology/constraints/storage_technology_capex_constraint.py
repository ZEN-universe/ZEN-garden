from zen_garden.elements.technology.masks import CapacityAdditionEligibleMask
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.registries.set_registry import SetRegistry
from zen_garden.utils import linexpr_from_tuple_np


class StorageTechnologyCapexConstraint(GenericConstraint):
    masks = {"constraint_storage_technology_capex": [CapacityAdditionEligibleMask]}

    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Definition of the capital expenditures for the storage technology.

        Formulation:

        .. math::
            C^{\\mathrm{cap,overnight}}_{h,n,y} =
            \\kappa^{\\mathrm{cap,power}}_{h,n,y}\\Delta K_{h,n,y}
            + \\kappa^{\\mathrm{cap,energy}}_{h,n,y}\\Delta K^{\\mathrm{energy}}_{h,n,y}

        The implementation stores the two terms separately along its capacity-type
        dimension; their sum is :math:`C^{\\mathrm{cap,overnight}}_{h,n,y}`.

        Notation:

        :math:`C^{\\mathrm{cap,overnight}}_{h,n,y}`: total overnight CAPEX of storage
        technology :math:`h` at node :math:`n` in year :math:`y`
        :math:`\\Delta K_{h,n,y}` and :math:`\\Delta K^{\\mathrm{energy}}_{h,n,y}`:
        power- and energy-capacity additions
        :math:`\\kappa^{\\mathrm{cap,power}}_{h,n,y}` and
        :math:`\\kappa^{\\mathrm{cap,energy}}_{h,n,y}`: specific power- and
        energy-capacity
        CAPEX
        """
        index_values, index_names = (
            model_constructor.optimization_model.create_custom_set(
                [
                    "set_storage_technologies",
                    "set_capacity_types",
                    "set_nodes",
                    "set_years",
                ]
            )
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
        # variables.coords merges all variables, so only compute it once
        variable_coords = model_constructor.optimization_model.lp_model.variables.coords
        coords = [
            variable_coords["set_storage_technologies"],
            variable_coords["set_capacity_types"],
            variable_coords["set_nodes"],
            variable_coords["set_years"],
        ]

        ### formulate constraint
        lhs = linexpr_from_tuple_np(
            [
                (
                    1.0,
                    model_constructor.optimization_model.variables[
                        "cost_capex_overnight"
                    ].loc[techs, capacity_types, nodes, times],
                ),
                (
                    -model_constructor.optimization_model.parameters.capex_specific_storage.loc[
                        techs, capacity_types, nodes, times
                    ],
                    model_constructor.optimization_model.variables[
                        "capacity_addition"
                    ].loc[techs, capacity_types, nodes, times],
                ),
            ],
            coords,
            model_constructor.optimization_model.lp_model,
        )
        rhs = 0
        constraints = lhs == rhs

        cls.add_constraint(
            model_constructor, "constraint_storage_technology_capex", constraints
        )
