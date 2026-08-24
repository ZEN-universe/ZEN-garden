import numpy as np
import xarray as xr

from zen_garden.elements.transport_technology import TransportTechnology
from zen_garden.model.components.set_registry import SetRegistry
from zen_garden.topology.generic_constraint import GenericConstraint


class TransportTechnologyCapexConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Definition of the capital expenditures for the transport technology.

        Formulation:

        .. math::
            \\text{if transport distance set to inf: } \\Delta K_{h,e,y} = 0
        .. math::
            \\text{else: } C^{\\mathrm{cap,overnight}}_{h,e,y} = \\Delta K_{h,e,y}
            \\kappa^{\\mathrm{cap,fixed}}_{h,y} +
            g_{h,e,y} d^{\\mathrm{dist}}_{h,e}
            \\kappa^{\\mathrm{cap,dist}}_{h,e,y}

        Notation:

        :math:`\\Delta K_{h,e,y}`: Capacity addition of transport technology :math:`h`
        on edge :math:`e` in year :math:`y`
        :math:`C^{\\mathrm{cap,overnight}}_{h,e,y}`: overnight CAPEX of transport
        technology :math:`h` on edge :math:`e` in year :math:`y`
        :math:`\\kappa^{\\mathrm{cap,fixed}}_{h,y}`: Specific constant capital
        expenditures of transport technology :math:`h` in year :math:`y`
        :math:`\\kappa^{\\mathrm{cap,dist}}_{h,e,y}`: Specific capital expenditures per
        distance of transport technology :math:`h` on edge :math:`e` in year :math:`y`
        :math:`g_{h,e,y}`: binary installation decision for
        transport technology :math:`h` on edge :math:`e` in year :math:`y`
        :math:`d^{\\mathrm{dist}}_{h,e}`: Transport distance for transport technology
        :math:`h` on edge :math:`e`
        """
        index_values, index_list = self.zen_model.create_custom_set(
            ["set_transport_technologies", "set_edges", "set_years"],
            TransportTechnology,
        )

        # check if we even need to continue
        if len(index_values) == 0:
            return
        # get the coords
        coords = [
            self.zen_model.parameters.capex_per_distance_transport.coords[
                "set_transport_technologies"
            ],
            self.zen_model.parameters.capex_per_distance_transport.coords["set_edges"],
            self.zen_model.parameters.capex_per_distance_transport.coords["set_years"],
        ]

        ### masks
        # This mask checks the distance between nodes for the condition
        mask = np.isinf(self.zen_model.parameters.distance).astype(float)

        # This mask ensure we only get constraints where we want them
        index_arrs = SetRegistry.tuple_to_arr(index_values, index_list)
        global_mask = xr.DataArray(False, coords=coords)
        global_mask.loc[index_arrs] = True

        ### auxiliary calculations TODO improve
        term_distance_inf = (
            mask
            * self.zen_model.variables["capacity_addition"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
        )
        term_distance_not_inf = (1 - mask) * (
            self.zen_model.variables["cost_capex_overnight"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
            - self.zen_model.variables["capacity_addition"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
            * self.zen_model.parameters.capex_specific_transport.loc[
                coords[0], coords[1]
            ]
        )
        # Additional check to avoid binary variables when their coefficient is 0
        if np.any(
            self.zen_model.parameters.distance.loc[coords[0], coords[1]]
            * self.zen_model.parameters.capex_per_distance_transport.loc[
                coords[0], coords[1]
            ]
            != 0
        ):
            term_distance_not_inf -= (
                (1 - mask)
                * self.zen_model.variables["technology_installation"].loc[
                    coords[0], "power", coords[1], coords[2]
                ]
                * (
                    self.zen_model.parameters.distance.loc[coords[0], coords[1]]
                    * self.zen_model.parameters.capex_per_distance_transport.loc[
                        coords[0], coords[1]
                    ]
                )
            )

        ### formulate constraint
        lhs = term_distance_inf + term_distance_not_inf
        lhs = lhs.where(global_mask)
        rhs = 0
        constraints = lhs == rhs
        self.zen_model.add_constraint(
            "constraint_transport_technology_capex", constraints
        )
