"""Rules for the TransportTechnology class."""

import logging
from typing import cast

import numpy as np
import xarray as xr

from zen_garden.elements.generic_rule import GenericRule
from zen_garden.model.components.index_set import IndexSet

logger = logging.getLogger(__name__)


class TransportTechnologyRules(GenericRule):
    """Rules for the TransportTechnology class."""

    def constraint_capacity_factor_transport(self):
        """Load is limited by the installed capacity and the maximum load factor.

        .. math::
            F_{j,e,t,y}^\\mathrm{r} \\leq m^{\\mathrm{max}}_{j,e,t,y}S_{j,e,y}


        :math:`F_{j,e,t,y}^\\mathrm{r}`: Reference flow of carrier through transport
        technology :math:`j` on edge :math:`i` and time :math:`t` in year :math:`y` \n
        :math:`m^{\\mathrm{max}}_{j,e,t,y}`: Maximum load factor of transport
        technology :math:`j` on edge :math:`i` and time :math:`t` in year :math:`y` \n
        :math:`S_{j,e,y}`: Capacity of transport technology :math:`j` on
        edge :math:`i` in year :math:`y`


        """
        techs = self.zen_model.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = self.zen_model.sets["set_edges"]
        times = self.zen_model.lp_model.variables["flow_transport"].coords[
            "set_time_steps_operation"
        ]
        time_step_year = xr.DataArray(
            [self.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
        )
        term_capacity = (
            self.zen_model.parameters.max_load.loc[techs, edges, :]
            * self.zen_model.lp_model.variables["capacity"].loc[
                techs, "power", edges, time_step_year
            ]
        ).rename(
            {
                "set_technologies": "set_transport_technologies",
                "set_location": "set_edges",
            }
        )

        lhs = (
            term_capacity
            - self.zen_model.lp_model.variables["flow_transport"].loc[techs, edges, :]
        )
        rhs = 0
        constraints = lhs >= rhs
        ### return
        self.zen_model.add_constraint(
            "constraint_capacity_factor_transport", constraints
        )

    def constraint_opex_emissions_technology_transport(self):
        """Calculate opex of each technology.

        .. math::
            O_{j,t,y}^\\mathrm{t} = \\beta_{j,y} F_{j,e,t,y}

        :math:`O_{h,p,t}^\\mathrm{t}`: Variable operating expenditures of transport
        technology :math:`j` on edge :math:`e` at time :math:`t` in year :math:`y` \n
        :math:`\\beta_{j,y}`: Specific variable operating expenditures of transport
        technology :math:`j` in year :math:`y` \n
        :math:`F_{j,e,t,y}`: Reference flow of carrier through transport
        technology :math:`j` on edge :math:`e` at time :math:`t` in year :math:`y`

        """
        techs = self.zen_model.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = self.zen_model.sets["set_edges"]
        lhs_opex = self.zen_model.lp_model.variables["cost_opex_variable"].loc[
            techs, edges, :
        ] - (
            self.zen_model.parameters.opex_specific_variable
            * self.zen_model.lp_model.variables["flow_transport"].rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
        ).sel(
            {"set_technologies": techs, "set_location": edges}
        )
        lhs_emissions = self.zen_model.lp_model.variables[
            "carbon_emissions_technology"
        ].loc[techs, edges, :] - (
            self.zen_model.parameters.carbon_intensity_technology
            * self.zen_model.lp_model.variables["flow_transport"].rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
        ).sel(
            {"set_technologies": techs, "set_location": edges}
        )
        lhs_opex = lhs_opex.rename(
            {
                "set_technologies": "set_transport_technologies",
                "set_location": "set_edges",
            }
        )
        lhs_emissions = lhs_emissions.rename(
            {
                "set_technologies": "set_transport_technologies",
                "set_location": "set_edges",
            }
        )
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs
        ### return
        self.zen_model.add_constraint(
            "constraint_opex_technology_transport", constraints_opex
        )
        self.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_transport", constraints_emissions
        )

    def constraint_transport_technology_losses_flow(self):
        """Compute the flow losses for a carrier through a transport technology.

        .. math::
            \\text{if transport distance set to inf: } F^\\mathrm{l}_{j,e,t} = 0
        .. math::
            \\text{else: } F^\\mathrm{l}_{j,e,t} = h_{j,e} \\rho_{j} F_{j,e,t}

        :math:`F^\\mathrm{l}_{j,e,t}`: Flow losses of carrier through transport
        technology :math:`j` on edge :math:`e` at time :math:`t` \n
        :math:`h_{j,e}`: Transport distance for transport technology :math:`j` on
        edge :math:`e` \n
        :math:`\\rho_{j}`: Loss factor for transport technology :math:`j` \n
        :math:`F_{j,e,t}`: Reference flow of carrier through transport
        technology :math:`j` on edge :math:`e` at time :math:`t`

        """
        if len(self.zen_model.sets["set_transport_technologies"]) == 0:
            return
        flow_transport = self.zen_model.lp_model.variables["flow_transport"]
        flow_transport_loss = self.zen_model.lp_model.variables["flow_transport_loss"]
        # This mask checks the distance between nodes
        distance_isfinite = cast(
            xr.DataArray, ~np.isinf(self.zen_model.parameters.distance)
        )
        mask = distance_isfinite.broadcast_like(flow_transport.lower)
        loss_factor = self.zen_model.parameters.transport_loss_factor.broadcast_like(
            flow_transport.lower
        )
        lhs = (flow_transport_loss - loss_factor * flow_transport).where(mask, 0)
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_transport_technology_losses_flow", constraints
        )

    def constraint_transport_technology_capex(self, index_values, index_list):
        """Definition of the capital expenditures for the transport technology.

        .. math::
            \\text{if transport distance set to inf: } \\Delta S_{j,e,y} = 0
        .. math::
            \\text{else: } CAPEX_{j,e,y} = \\Delta S_{j,e,y}
            \\alpha_{j,y}^{\\mathrm{const}} +
            \\Delta S_{j,e,y} h_{j,e} \\alpha^\\mathrm{dist}_{j,e,y}

        :math:`\\Delta S_{j,e,y}`: Capacity addition of transport technology :math:`j`
        on edge :math:`e` in year :math:`y` \n
        :math:`CAPEX_{j,e,y}`: Capital expenditures of transport technology :math:`j`
        on edge :math:`e` in year :math:`y` \n
        :math:`\\alpha_{j,y}^{\\mathrm{const}}`: Specific constant capital expenditures
        of transport technology :math:`j` in year :math:`y`
        :math:`\\alpha^\\mathrm{dist}_{j,e,y}`: Specific capital expenditures per
        distance of transport technology :math:`j` on edge :math:`e` in year :math:`y`
        :math:`h_{j,e}`: Transport distance for transport technology :math:`j` on
        edge :math:`e`

        """
        # check if we even need to continue
        if len(index_values) == 0:
            return []
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
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        global_mask = xr.DataArray(False, coords=coords)
        global_mask.loc[index_arrs] = True

        ### auxiliary calculations TODO improve
        term_distance_inf = (
            mask
            * self.zen_model.lp_model.variables["capacity_addition"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
        )
        term_distance_not_inf = (1 - mask) * (
            self.zen_model.lp_model.variables["cost_capex_overnight"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
            - self.zen_model.lp_model.variables["capacity_addition"].loc[
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
                * self.zen_model.lp_model.variables["technology_installation"].loc[
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
