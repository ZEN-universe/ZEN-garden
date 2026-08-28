import numpy as np
import xarray as xr

from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.registries.set_registry import SetRegistry


class TransportTechnologyCapexConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Definition of the capital expenditures for the transport technology.

        Formulation:

        .. math::
            \\text{if transport distance set to inf: } \\Delta K_{h,e,y} = 0

        .. math::
            \\text{else: } C^{\\mathrm{cap,overnight}}_{h,e,y} =
            \\Delta K_{h,e,y}\\kappa^{\\mathrm{cap,fixed}}_{h,e,y}
            + g_{h,e,y}d^{\\mathrm{dist}}_{h,e}
            \\kappa^{\\mathrm{cap,dist}}_{h,e,y}

        Notation:

        :math:`\\Delta K_{h,e,y}`: Capacity addition of transport technology :math:`h`
        on edge :math:`e` in year :math:`y`
        :math:`C^{\\mathrm{cap,overnight}}_{h,e,y}`: overnight CAPEX of transport
        technology :math:`h` on edge :math:`e` in year :math:`y`
        :math:`\\kappa^{\\mathrm{cap,fixed}}_{h,e,y}`: Specific constant capital
        expenditures of transport technology :math:`h` in year :math:`y`
        :math:`\\kappa^{\\mathrm{cap,dist}}_{h,e,y}`: Specific capital expenditures per
        distance of transport technology :math:`h` on edge :math:`e` in year :math:`y`
        :math:`g_{h,e,y}`: binary installation decision for
        transport technology :math:`h` on edge :math:`e` in year :math:`y`
        :math:`d^{\\mathrm{dist}}_{h,e}`: Transport distance for transport technology
        :math:`h` on edge :math:`e`
        """
        index_values, index_list = model_constructor.zen_model.create_custom_set(
            ["set_transport_technologies", "set_edges", "set_years"]
        )

        # check if we even need to continue
        if len(index_values) == 0:
            return
        # get the coords
        coords = [
            model_constructor.zen_model.parameters.capex_per_distance_transport.coords[
                "set_transport_technologies"
            ],
            model_constructor.zen_model.parameters.capex_per_distance_transport.coords[
                "set_edges"
            ],
            model_constructor.zen_model.parameters.capex_per_distance_transport.coords[
                "set_years"
            ],
        ]

        ### masks
        # This mask checks the distance between nodes for the condition
        mask = np.isinf(model_constructor.zen_model.parameters.distance).astype(float)

        # This mask ensure we only get constraints where we want them
        index_arrs = SetRegistry.tuple_to_arr(index_values, index_list)
        global_mask = xr.DataArray(False, coords=coords)
        global_mask.loc[index_arrs] = True

        ### auxiliary calculations TODO improve
        term_distance_inf = (
            mask
            * model_constructor.zen_model.variables["capacity_addition"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
        )
        term_distance_not_inf = (1 - mask) * (
            model_constructor.zen_model.variables["cost_capex_overnight"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
            - model_constructor.zen_model.variables["capacity_addition"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
            * model_constructor.zen_model.parameters.capex_specific_transport.loc[
                coords[0], coords[1]
            ]
        )
        # Additional check to avoid binary variables when their coefficient is 0
        if np.any(
            model_constructor.zen_model.parameters.transport_capex_distance.loc[
                coords[0], coords[1]
            ]
            != 0
        ):
            term_distance_not_inf -= (
                (1 - mask)
                * model_constructor.zen_model.variables["technology_installation"].loc[
                    coords[0], "power", coords[1], coords[2]
                ]
                * (
                    model_constructor.zen_model.parameters.transport_capex_distance.loc[
                        coords[0], coords[1]
                    ]
                )
            )

        ### formulate constraint
        lhs = term_distance_inf + term_distance_not_inf
        lhs = lhs.where(global_mask)
        rhs = 0
        constraints = lhs == rhs
        model_constructor.zen_model.add_constraint(
            "constraint_transport_technology_capex", constraints
        )
