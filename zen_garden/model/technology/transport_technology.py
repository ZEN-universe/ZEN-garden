"""Class defining the parameters, variables and constraints that hold for all transport
technologies. The class takes the abstract optimization model as an input, and returns
the parameters, variables and constraints that hold for the transport technologies.
"""

import logging
from typing import TYPE_CHECKING, override

import numpy as np
import xarray as xr

from zen_garden.model.components.index_set import IndexSet
from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.element import ElementConstructor
from zen_garden.model.generic_rule import GenericRule
from zen_garden.model.technology import Technology
from zen_garden.model.zen_model import ZenModel

if TYPE_CHECKING:
    from zen_garden.model.energy_system import EnergySystem
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class TransportTechnology(Technology):
    # set label
    label = "set_transport_technologies"
    location_type = "set_edges"

    def __init__(
        self,
        tech: str,
        config: Config,
        context: Context,
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
        unit_handling: "UnitHandling",
    ):
        """Init transport technology object.

        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(
            tech, config, context, energy_system, element_registry, unit_handling
        )
        # dict of reversed edges
        self.dict_reversed_edges = {}
        # store carriers of transport technology
        self.store_carriers()

    def store_carriers(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().store_carriers()

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # get attributes from class <Technology>
        super().store_input_data()
        # set attributes for parameters of child class <TransportTechnology>
        self.distance = self.data_input.extract_input_data(
            "distance", index_sets=["set_edges"], unit_category={"distance": 1}
        )
        if "/ kilometer" in str(
            self.units["carbon_intensity_technology"]["unit_in_base_units"].units
        ):
            self.carbon_intensity_technology = self.data_input.extract_input_data(
                "carbon_intensity_technology",
                index_sets=["set_edges"],
                unit_category={"emissions": 1, "energy_quantity": -1, "distance": -1},
            )
            self.carbon_intensity_technology *= self.distance
        # get transport loss factor
        self.get_transport_loss_factor()
        # get capex of transport technology
        self.get_capex_transport()
        # annualize capex
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def get_transport_loss_factor(self):
        """Get transport loss factor."""
        # check which transport loss factor is used
        assert not (
            "transport_loss_factor_linear" in self.data_input.attribute_dict
            and "transport_loss_factor_exponential" in self.data_input.attribute_dict
        ), "Only one transport loss factor can be specified."
        if "transport_loss_factor_linear" in self.data_input.attribute_dict:
            self.transport_loss_factor = self.data_input.extract_input_data(
                "transport_loss_factor_linear",
                index_sets=[],
                unit_category={"distance": -1},
            )
            self.transport_loss_factor = self.transport_loss_factor[0] * self.distance
        elif "transport_loss_factor_exponential" in self.data_input.attribute_dict:
            self.transport_loss_factor = self.data_input.extract_input_data(
                "transport_loss_factor_exponential",
                index_sets=[],
                unit_category={"distance": -1},
            )
            self.transport_loss_factor = 1 - np.exp(
                -self.transport_loss_factor[0] * self.distance
            )
            self.config.system.set_transport_technologies_loss_exponential.append(
                self.name
            )
        else:
            raise AttributeError(
                f"The transport technology {self.name} has neither of the attributes: "
                f"transport_loss_factor_linear nor transport_loss_factor_exponential."
            )

    def get_capex_transport(self):
        """Get capex of transport technology."""
        # check if there are separate capex for capacity and distance
        if self.config.system.double_capex_transport:
            # both capex terms must be specified
            self.capex_specific_transport = self.data_input.extract_input_data(
                "capex_specific_transport",
                index_sets=["set_edges", "set_time_steps_yearly"],
                time_steps="set_time_steps_yearly",
                unit_category={"money": 1, "energy_quantity": -1, "time": 1},
            )
            self.capex_per_distance_transport = self.data_input.extract_input_data(
                "capex_per_distance_transport",
                index_sets=["set_edges", "set_time_steps_yearly"],
                time_steps="set_time_steps_yearly",
                unit_category={"money": 1, "distance": -1},
            )
        else:
            # Only capex_specific is used, capex_per_distance_transport is set to Zero.
            if "capex_per_distance_transport" in self.data_input.attribute_dict:
                self.capex_per_distance_transport = self.data_input.extract_input_data(
                    "capex_per_distance_transport",
                    index_sets=["set_edges", "set_time_steps_yearly"],
                    time_steps="set_time_steps_yearly",
                    unit_category={
                        "money": 1,
                        "distance": -1,
                        "energy_quantity": -1,
                        "time": 1,
                    },
                )
                self.capex_specific_transport = (
                    self.capex_per_distance_transport * self.distance
                )
            elif "capex_specific_transport" in self.data_input.attribute_dict:
                self.capex_specific_transport = self.data_input.extract_input_data(
                    "capex_specific_transport",
                    index_sets=["set_edges", "set_time_steps_yearly"],
                    time_steps="set_time_steps_yearly",
                    unit_category={"money": 1, "energy_quantity": -1, "time": 1},
                )
            else:
                raise AttributeError(
                    f"The transport technology {self.name} has neither "
                    f"capex_per_distance_transport nor capex_specific_transport "
                    f"attribute."
                )
            self.capex_per_distance_transport = self.capex_specific_transport * 0.0
        if "opex_specific_fixed_per_distance" in self.data_input.attribute_dict:
            self.opex_specific_fixed_per_distance = self.data_input.extract_input_data(
                "opex_specific_fixed_per_distance",
                index_sets=["set_edges", "set_time_steps_yearly"],
                unit_category={
                    "money": 1,
                    "distance": -1,
                    "energy_quantity": -1,
                    "time": 1,
                },
            )
            self.opex_specific_fixed = (
                self.opex_specific_fixed_per_distance * self.distance
            )
        elif "opex_specific_fixed" in self.data_input.attribute_dict:
            self.opex_specific_fixed = self.data_input.extract_input_data(
                "opex_specific_fixed",
                index_sets=["set_edges", "set_time_steps_yearly"],
                time_steps="set_time_steps_yearly",
                unit_category={"money": 1, "energy_quantity": -1, "time": 1},
            )
        else:
            raise AttributeError(
                f"The transport technology {self.name} has neither "
                f"opex_specific_fixed_per_distance nor opex_specific_fixed attribute."
            )

    def convert_to_fraction_of_capex(self):
        """Converts total capex to fraction of capex.

        this method converts the total capex to fraction of capex, depending on how
        many hours per year are calculated.
        """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.capex_specific_transport = self.capex_specific_transport * fraction_year
        self.capex_per_distance_transport = (
            self.capex_per_distance_transport * fraction_year
        )

    def calculate_capex_of_single_capacity(self, capacity, index, **kwargs):
        """This method calculates the capex of a single existing capacity.

        :param capacity: capacity of transport technology
        :param index: index of capacity
        :return: capex of single capacity
        """
        if np.isnan(self.capex_specific_transport[index[0]].iloc[0]) and np.isnan(
            self.capex_per_distance_transport[index[0]].iloc[0]
        ):
            return 0
        elif self.energy_system.config.system.double_capex_transport and capacity != 0:
            return (
                self.capex_specific_transport[index[0]].iloc[0] * capacity
                + self.capex_per_distance_transport[index[0]].iloc[0]
                * self.distance[index[0]]
            )
        else:
            return self.capex_specific_transport[index[0]].iloc[0] * capacity

    ### --- getter/setter classmethods
    def set_reversed_edge(self, edge, reversed_edge):
        """Maps the reversed edge to an edge.

        :param edge: edge
        :param reversed_edge: reversed edge
        """
        self.dict_reversed_edges[edge] = reversed_edge

    def get_reversed_edge(self, edge):
        """Get the reversed edge corresponding to an edge.

        :param edge: edge
        :return: reversed edge
        """
        return self.dict_reversed_edges[edge]


class TransportTechnologyConstructor(ElementConstructor):
    element_class = TransportTechnology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class <Carrier>.

        :return: True if there are elements, False otherwise
        """
        return True

    def construct_sets(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Sets of the class <TransportTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing sets for TransportTechnology")

    def construct_params(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Params of the class <TransportTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing parameters for TransportTechnology")

        # distance between nodes
        self.add_parameter(
            zen_model,
            energy_system,
            name="distance",
            index_names=["set_transport_technologies", "set_edges"],
            doc="distance between two nodes for transport technologies",
        )
        # capital cost per unit
        self.add_parameter(
            zen_model,
            energy_system,
            name="capex_specific_transport",
            index_names=[
                "set_transport_technologies",
                "set_edges",
                "set_time_steps_yearly",
            ],
            doc="capex per unit for transport technologies",
        )
        # capital cost per distance
        self.add_parameter(
            zen_model,
            energy_system,
            name="capex_per_distance_transport",
            index_names=[
                "set_transport_technologies",
                "set_edges",
                "set_time_steps_yearly",
            ],
            doc="capex per distance for transport technologies",
        )
        # carrier losses
        self.add_parameter(
            zen_model,
            energy_system,
            name="transport_loss_factor",
            index_names=["set_transport_technologies", "set_edges"],
            doc="linear carrier losses due to transport with transport technologies",
        )

    def construct_vars(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Vars of the class <TransportTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing variables for TransportTechnology")

        model = zen_model.lp_model
        variables = zen_model.variables
        sets = zen_model.sets

        def flow_transport_bounds(index_values, index_list):
            """Return bounds of carrier_flow for bigM expression.

            :param index_values: list of tuples with the index values
            :param index_list: The names of the indices
            :return bounds: bounds of carrier_flow
            """
            # get the arrays
            tech_arr, edge_arr, time_arr = sets.tuple_to_arr(index_values, index_list)
            # convert operationTimeStep to time_step_year:
            #   operationTimeStep -> base_time_step -> time_step_year
            ts = energy_system.time_steps
            time_step_year = xr.DataArray(
                [ts.convert_time_step_operation2year(time) for time in time_arr.data]
            )

            lower = (
                model.variables["capacity"]
                .lower.loc[tech_arr, "power", edge_arr, time_step_year]
                .data
            )
            upper = (
                model.variables["capacity"]
                .upper.loc[tech_arr, "power", edge_arr, time_step_year]
                .data
            )
            return np.stack([lower, upper], axis=-1)

        # flow of carrier on edge
        index_values, index_names = self.create_custom_set(
            ["set_transport_technologies", "set_edges", "set_time_steps_operation"],
            zen_model,
            energy_system,
        )
        bounds = flow_transport_bounds(index_values, index_names)
        variables.add_variable(
            name="flow_transport",
            index_sets=(index_values, index_names),
            bounds=bounds,
            doc="carrier flow through transport technology on edge i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # loss of carrier on edge
        variables.add_variable(
            name="flow_transport_loss",
            index_sets=(index_values, index_names),
            bounds=(0, np.inf),
            doc="carrier flow lost due to resistances etc. by transporting carrier "
            "through transport technology on edge i and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )

    def construct_constraints(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Constraints of the class <TransportTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing constraints for TransportTechnology")

        rules = TransportTechnologyRules(self.config, self.context)

        # limit flow by capacity and max load
        rules.constraint_capacity_factor_transport(zen_model, energy_system)

        # opex and emissions constraint for transport technologies
        rules.constraint_opex_emissions_technology_transport(zen_model, energy_system)

        # carrier flow Losses
        rules.constraint_transport_technology_losses_flow(zen_model, energy_system)

        # capex of transport technologies
        index_values, index_list = self.create_custom_set(
            ["set_transport_technologies", "set_edges", "set_time_steps_yearly"],
            zen_model,
            energy_system,
        )
        rules.constraint_transport_technology_capex(
            zen_model, energy_system, index_values, index_list
        )


class TransportTechnologyRules(GenericRule):
    """Rules for the TransportTechnology class."""

    def __init__(self, config: Config, context: Context):
        """Inits the rules for a given ".

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(config, context)

    def constraint_capacity_factor_transport(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
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
        techs = zen_model.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = zen_model.sets["set_edges"]
        times = zen_model.lp_model.variables["flow_transport"].coords[
            "set_time_steps_operation"
        ]
        ts = energy_system.time_steps
        time_step_year = xr.DataArray(
            [ts.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
        )
        term_capacity = (
            zen_model.parameters.max_load.loc[techs, edges, :]
            * zen_model.lp_model.variables["capacity"].loc[
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
            - zen_model.lp_model.variables["flow_transport"].loc[techs, edges, :]
        )
        rhs = 0
        constraints = lhs >= rhs
        ### return
        zen_model.constraints.add_constraint(
            "constraint_capacity_factor_transport", constraints
        )

    def constraint_opex_emissions_technology_transport(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
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
        techs = zen_model.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = zen_model.sets["set_edges"]
        lhs_opex = zen_model.lp_model.variables["cost_opex_variable"].loc[
            techs, edges, :
        ] - (
            zen_model.parameters.opex_specific_variable
            * zen_model.lp_model.variables["flow_transport"].rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
        ).sel(
            {"set_technologies": techs, "set_location": edges}
        )
        lhs_emissions = zen_model.lp_model.variables["carbon_emissions_technology"].loc[
            techs, edges, :
        ] - (
            zen_model.parameters.carbon_intensity_technology
            * zen_model.lp_model.variables["flow_transport"].rename(
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
        zen_model.constraints.add_constraint(
            "constraint_opex_technology_transport", constraints_opex
        )
        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_technology_transport", constraints_emissions
        )

    def constraint_transport_technology_losses_flow(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
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
        if len(zen_model.sets["set_transport_technologies"]) == 0:
            return
        flow_transport = zen_model.lp_model.variables["flow_transport"]
        flow_transport_loss = zen_model.lp_model.variables["flow_transport_loss"]
        # This mask checks the distance between nodes
        mask = (~np.isinf(zen_model.parameters.distance)).broadcast_like(
            flow_transport.lower
        )
        loss_factor = zen_model.parameters.transport_loss_factor.broadcast_like(
            flow_transport.lower
        )
        lhs = (flow_transport_loss - loss_factor * flow_transport).where(mask, 0.0)
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_transport_technology_losses_flow", constraints
        )

    def constraint_transport_technology_capex(
        self,
        zen_model: ZenModel,
        energy_system: "EnergySystem",
        index_values,
        index_list,
    ):
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
            zen_model.parameters.capex_per_distance_transport.coords[
                "set_transport_technologies"
            ],
            zen_model.parameters.capex_per_distance_transport.coords["set_edges"],
            zen_model.parameters.capex_per_distance_transport.coords[
                "set_time_steps_yearly"
            ],
        ]

        ### masks
        # This mask checks the distance between nodes for the condition
        mask = np.isinf(zen_model.parameters.distance).astype(float)

        # This mask ensure we only get constraints where we want them
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        global_mask = xr.DataArray(False, coords=coords)
        global_mask.loc[index_arrs] = True

        ### auxiliary calculations TODO improve
        term_distance_inf = (
            mask
            * zen_model.lp_model.variables["capacity_addition"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
        )
        term_distance_not_inf = (1 - mask) * (
            zen_model.lp_model.variables["cost_capex_overnight"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
            - zen_model.lp_model.variables["capacity_addition"].loc[
                coords[0], "power", coords[1], coords[2]
            ]
            * zen_model.parameters.capex_specific_transport.loc[coords[0], coords[1]]
        )
        # Additional check to avoid binary variables when their coefficient is 0
        if np.any(
            zen_model.parameters.distance.loc[coords[0], coords[1]]
            * zen_model.parameters.capex_per_distance_transport.loc[
                coords[0], coords[1]
            ]
            != 0
        ):
            term_distance_not_inf -= (
                (1 - mask)
                * zen_model.lp_model.variables["technology_installation"].loc[
                    coords[0], "power", coords[1], coords[2]
                ]
                * (
                    zen_model.parameters.distance.loc[coords[0], coords[1]]
                    * zen_model.parameters.capex_per_distance_transport.loc[
                        coords[0], coords[1]
                    ]
                )
            )

        ### formulate constraint
        lhs = term_distance_inf + term_distance_not_inf
        lhs = lhs.where(global_mask)
        rhs = 0
        constraints = lhs == rhs
        zen_model.constraints.add_constraint(
            "constraint_transport_technology_capex", constraints
        )
