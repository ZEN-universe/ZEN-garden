"""Class defining a standard EnergySystem.
Contains methods to construct the energy system from the given input data and that
defines the variables, parameters and constraints which apply to the Energy System.
The class takes the abstract optimization model as an input.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.generic_rule import GenericRule
from zen_garden.model.time_steps import TimeStepsDicts
from zen_garden.model.zen_model import ZenModel
from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.preprocess.unit_handling import UnitHandling

logger = logging.getLogger(__name__)


class EnergySystem:
    """Class defining a standard energy system."""

    def __init__(
        self,
        config: Config,
        context: Context,
        unit_handling: UnitHandling,
    ):
        """Initialization of the energy_system.

        :param optimization_setup: The OptimizationSetup of the EnergySystem class
        :param config: The Config of the entire setup
        :param context: The Context of the entire setup
        """
        # the name
        self.name = "EnergySystem"
        self._name = "EnergySystem"
        # set attributes
        self.config = config
        self.context = context
        self.unit_handling = unit_handling

        # empty dict of technologies of carrier
        self.dict_technology_of_carrier = {}
        # The timesteps
        self.time_steps = TimeStepsDicts()

        # empty list of indexing sets
        self.input_path = Path(self.context.paths["energy_system"]["folder"])
        self.indexing_sets = [key for key in self.config.system.keys() if "set" in key]

        # create DataInput object
        self.data_input = DataInput(
            element=self,
            energy_system=self,
            unit_handling=self.unit_handling,
            config=config,
            context=context,
        )
        # initialize empty set_carriers list
        self.set_carriers = []
        # dict to save the parameter units (and save them in the results later on)
        self.units = {}

    def store_input_data(self):
        """Retrieves and stores input data for EnergySystem as attributes."""
        # store scenario dict
        self.data_input.scenario_dict = self.context.scenario_dict
        # in class <EnergySystem>, all sets are constructed
        self.set_nodes = self.data_input.extract_locations()
        self.set_nodes_on_edges = self.calculate_edges_from_nodes()
        self.set_edges = list(self.set_nodes_on_edges.keys())
        self.set_haversine_distances_edges = (
            self.calculate_haversine_distances_from_nodes()
        )
        self.set_technologies = self.config.system.set_technologies
        # base time steps
        self.set_base_time_steps = list(
            range(
                0,
                self.config.system.unaggregated_time_steps_per_year
                * self.config.system.optimized_years,
            )
        )
        self.set_base_time_steps_yearly = list(
            range(0, self.config.system.unaggregated_time_steps_per_year)
        )

        # yearly time steps
        self.set_time_steps_yearly = list(range(self.config.system.optimized_years))
        self.set_time_steps_yearly_entire_horizon = copy.deepcopy(
            self.set_time_steps_yearly
        )
        time_steps_yearly_duration = self.time_steps.calculate_time_step_duration(
            self.set_time_steps_yearly, self.set_base_time_steps
        )
        self.sequence_time_steps_yearly = np.concatenate(
            [
                [time_step] * time_steps_yearly_duration[time_step]
                for time_step in time_steps_yearly_duration
            ]
        )
        self.time_steps.sequence_time_steps_yearly = self.sequence_time_steps_yearly
        # list containing simulated years
        #   (needed for convert_real_to_generic_time_indices() in extract_input_data.py)
        self.set_time_steps_years = list(
            range(
                self.config.system.reference_year,
                self.config.system.reference_year
                + self.config.system.optimized_years
                * self.config.system.interval_between_years,
                self.config.system.interval_between_years,
            )
        )
        """ parameters whose time-dependant data should not be interpolated
            (for years without data) in the extract_input_data.py
            convert_real_to_generic_time_indices() function"""
        self.parameters_interpolation_off = self.data_input.read_input_json(
            "parameters_interpolation_off"
        )
        # technology-specific
        self.set_conversion_technologies = (
            self.config.system.set_conversion_technologies
        )
        self.set_transport_technologies = self.config.system.set_transport_technologies
        self.set_storage_technologies = self.config.system.set_storage_technologies
        self.set_retrofitting_technologies = (
            self.config.system.set_retrofitting_technologies
        )
        # discount rate
        self.discount_rate = self.data_input.extract_input_data(
            "discount_rate", index_sets=[], unit_category={}
        )
        # carbon emissions limit
        self.carbon_emissions_annual_limit = self.data_input.extract_input_data(
            "carbon_emissions_annual_limit",
            index_sets=["set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"emissions": 1},
        )
        _fraction_year = (
            self.config.system.unaggregated_time_steps_per_year
            / self.config.system.total_hours_per_year
        )
        self.carbon_emissions_annual_limit = (
            self.carbon_emissions_annual_limit * _fraction_year
        )  # reduce to fraction of year
        self.carbon_emissions_budget = self.data_input.extract_input_data(
            "carbon_emissions_budget", index_sets=[], unit_category={"emissions": 1}
        )
        self.carbon_emissions_cumulative_existing = self.data_input.extract_input_data(
            "carbon_emissions_cumulative_existing",
            index_sets=[],
            unit_category={"emissions": 1},
        )
        # price carbon emissions
        self.price_carbon_emissions = self.data_input.extract_input_data(
            "price_carbon_emissions",
            index_sets=["set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"money": 1, "emissions": -1},
        )
        self.price_carbon_emissions_budget_overshoot = (
            self.data_input.extract_input_data(
                "price_carbon_emissions_budget_overshoot",
                index_sets=[],
                unit_category={"money": 1, "emissions": -1},
            )
        )
        self.price_carbon_emissions_annual_overshoot = (
            self.data_input.extract_input_data(
                "price_carbon_emissions_annual_overshoot",
                index_sets=[],
                unit_category={"money": 1, "emissions": -1},
            )
        )
        # market share unbounded
        self.market_share_unbounded = self.data_input.extract_input_data(
            "market_share_unbounded", index_sets=[], unit_category={}
        )
        # knowledge_spillover_rate
        self.knowledge_depreciation_rate = self.data_input.extract_input_data(
            "knowledge_depreciation_rate", index_sets=[], unit_category={}
        )
        self.knowledge_spillover_rate = self.data_input.extract_input_data(
            "knowledge_spillover_rate", index_sets=[], unit_category={}
        )

    def calculate_edges_from_nodes(self):
        """Calculates set_nodes_on_edges from set_nodes.

        :return: set_nodes_on_edges: dict with edges and corresponding nodes
        """
        set_nodes_on_edges = {}
        # read edge file
        set_edges_input = self.data_input.extract_locations(extract_nodes=False)
        for edge in set_edges_input.index:
            set_nodes_on_edges[edge] = (
                set_edges_input.loc[edge, "node_from"],
                set_edges_input.loc[edge, "node_to"],
            )
        return set_nodes_on_edges

    def calculate_haversine_distances_from_nodes(self):
        """Computes the distance (in km) between two nodes.

        The Haversine function is used to compute the distance in kilometers based on
         their lon lat coordinates.

        :return: dict containing all edges along with their distances
        """
        set_haversine_distances_of_edges = {}

        # read coords file
        df_coords_input = self.data_input.extract_locations(extract_coordinates=True)
        if not isinstance(df_coords_input, pd.DataFrame):
            raise TypeError(
                "[EnergySystem] df_coords_input is not of type pd.DataFrame"
            )
        coords = df_coords_input.set_index("node")
        # TODO: load this outside this function
        self.config.system.coords = coords.T.to_dict()

        # convert coords from decimal degrees to radians
        df_coords_input["lon"] = df_coords_input["lon"] * np.pi / 180
        df_coords_input["lat"] = df_coords_input["lat"] * np.pi / 180
        # Radius of the Earth in kilometers
        radius = 6371.0
        for edge, nodes in self.set_nodes_on_edges.items():
            node_1, node_2 = nodes
            coords1 = df_coords_input[df_coords_input["node"] == node_1]
            coords2 = df_coords_input[df_coords_input["node"] == node_2]
            # Haversine formula
            dlon = coords2["lon"].squeeze() - coords1["lon"].squeeze()
            dlat = coords2["lat"].squeeze() - coords1["lat"].squeeze()
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(coords1["lat"].squeeze())
                * np.cos(coords2["lat"].squeeze())
                * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = radius * c
            set_haversine_distances_of_edges[edge] = distance
        multiplier = self.unit_handling.get_unit_multiplier(
            "km", attribute_name="distance"
        )
        set_haversine_distances_of_edges = {
            key: value * multiplier
            for key, value in set_haversine_distances_of_edges.items()
        }
        return set_haversine_distances_of_edges

    def set_technology_of_carrier(self, technology, list_technology_of_carrier):
        """Appends technology to carrier in dict_technology_of_carrier.

        :param technology: name of technology in model
        :param list_technology_of_carrier: list of carriers correspondent to technology
        """
        for carrier in list_technology_of_carrier:
            if carrier not in self.dict_technology_of_carrier:
                self.dict_technology_of_carrier[carrier] = [technology]
                self.set_carriers.append(carrier)
            elif technology not in self.dict_technology_of_carrier[carrier]:
                self.dict_technology_of_carrier[carrier].append(technology)

    def calculate_connected_edges(self, node, direction: str):
        """Calculates connected edges going in or going out.

        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode,
            out: node = startnode
        :return: _set_connected_edges: list of connected edges
        """
        if direction == "in":
            # second entry is node into which the flow goes
            _set_connected_edges = [
                edge
                for edge in self.set_nodes_on_edges
                if self.set_nodes_on_edges[edge][1] == node
            ]
        elif direction == "out":
            # first entry is node out of which the flow starts
            _set_connected_edges = [
                edge
                for edge in self.set_nodes_on_edges
                if self.set_nodes_on_edges[edge][0] == node
            ]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return _set_connected_edges

    def calculate_reversed_edge(self, edge):
        """Calculates the reversed edge corresponding to an edge.

        :param edge: input edge
        :return: _reversed_edge: edge corresponding to the reversed direction of edge
        """
        _node_out, _node_in = self.set_nodes_on_edges[edge]
        for _reversed_edge in self.set_nodes_on_edges:
            if (
                _node_out == self.set_nodes_on_edges[_reversed_edge][1]
                and _node_in == self.set_nodes_on_edges[_reversed_edge][0]
            ):
                return _reversed_edge
        raise KeyError(
            f"Edge {edge} has no reversed edge. "
            f"However, at least one transport technology is bidirectional"
        )

    ### --- methods to construct sets, parameters, variables, and constraints, that
    # correspond to EnergySystem --- ###

    def construct_sets(self, zen_model: ZenModel):
        """Constructs the pe.Sets of the class <EnergySystem>."""
        logger.info("Constructing sets for EnergySystem")

        # construct pe.Sets of the class <EnergySystem>
        # nodes
        zen_model.sets.add_set(
            name="set_nodes", data=self.set_nodes, doc="Set of nodes"
        )
        # edges
        zen_model.sets.add_set(
            name="set_edges", data=self.set_edges, doc="Set of edges"
        )
        # nodes on edges
        zen_model.sets.add_set(
            name="set_nodes_on_edges",
            data=self.set_nodes_on_edges,
            doc="Set of nodes that constitute an edge. "
            "Edge connects first node with second node.",
            index_set="set_edges",
        )
        # carriers
        zen_model.sets.add_set(
            name="set_carriers", data=self.set_carriers, doc="Set of carriers"
        )
        # technologies
        zen_model.sets.add_set(
            name="set_technologies", data=self.set_technologies, doc="set_technologies"
        )
        # all elements
        data = list(
            set(zen_model.sets["set_technologies"])
            | set(zen_model.sets["set_carriers"])
        )
        zen_model.sets.add_set(name="set_elements", data=data, doc="Set of elements")
        # set set_elements to indexing_sets
        self.indexing_sets.append("set_elements")
        # time-steps
        zen_model.sets.add_set(
            name="set_base_time_steps",
            data=self.set_base_time_steps,
            doc="Set of base time-steps",
        )
        # yearly time steps
        zen_model.sets.add_set(
            name="set_time_steps_yearly",
            data=self.set_time_steps_yearly,
            doc="Set of yearly time-steps",
        )
        # yearly time steps of entire optimization horizon
        zen_model.sets.add_set(
            name="set_time_steps_yearly_entire_horizon",
            data=self.set_time_steps_yearly_entire_horizon,
            doc="Set of yearly time-steps of entire optimization horizon",
        )
        # operational time steps
        zen_model.sets.add_set(
            name="set_time_steps_operation",
            data=self.time_steps.time_steps_operation,
            doc="Set of operational time steps",
        )
        # storage time steps
        zen_model.sets.add_set(
            name="set_time_steps_storage",
            data=self.time_steps.time_steps_storage,
            doc="Set of storage level time steps",
        )

    def construct_params(self, zen_model: ZenModel):
        """Constructs the pe.Params of the class <EnergySystem>."""
        logger.info("Constructing parameters for EnergySystem")

        # operational time step duration
        self._add_parameter(
            zen_model,
            name="time_steps_operation_duration",
            set_time_steps="set_time_steps_operation",
            doc="Parameter which specifies the duration of each operational time step",
        )
        # storage time step duration
        self._add_parameter(
            zen_model,
            name="time_steps_storage_duration",
            set_time_steps="set_time_steps_storage",
            doc="Parameter which specifies the duration of each storage time step",
        )
        # discount rate
        self._add_parameter(
            zen_model,
            name="discount_rate",
            doc="Parameter which specifies the discount rate of the energy system",
        )
        # carbon emissions limit
        self._add_parameter(
            zen_model,
            name="carbon_emissions_annual_limit",
            set_time_steps="set_time_steps_yearly",
            doc="Parameter which specifies the total limit on carbon emissions",
        )
        # carbon emissions budget
        self._add_parameter(
            zen_model,
            name="carbon_emissions_budget",
            doc="Parameter which specifies the total budget of carbon emissions "
            "until the end of the entire time horizon",
        )
        # carbon emissions budget
        self._add_parameter(
            zen_model,
            name="carbon_emissions_cumulative_existing",
            doc="Parameter which specifies the total previous carbon emissions",
        )
        # carbon price
        self._add_parameter(
            zen_model,
            name="price_carbon_emissions",
            set_time_steps="set_time_steps_yearly",
            doc="Parameter which specifies the yearly carbon price",
        )
        # carbon price of budget overshoot
        self._add_parameter(
            zen_model,
            name="price_carbon_emissions_budget_overshoot",
            doc="Parameter which specifies the carbon price for budget overshoot",
        )
        # carbon price of annual overshoot
        self._add_parameter(
            zen_model,
            name="price_carbon_emissions_annual_overshoot",
            doc="Parameter which specifies the carbon price for annual overshoot",
        )
        # carbon price of overshoot
        self._add_parameter(
            zen_model,
            name="market_share_unbounded",
            doc="Parameter which specifies the unbounded market share",
        )
        # knowledge depreciation rate
        self._add_parameter(
            zen_model,
            name="knowledge_depreciation_rate",
            doc="Parameter which specifies the knowledge depreciation rate",
        )
        # knowledge spillover rate
        self._add_parameter(
            zen_model,
            name="knowledge_spillover_rate",
            doc="Parameter which specifies the knowledge spillover rate",
        )

    def construct_vars(self, zen_model: ZenModel):
        """Constructs the pe.Vars of the class <EnergySystem>."""
        logger.info("Constructing variables for EnergySystem")

        # carbon emissions
        zen_model.variables.add_variable(
            name="carbon_emissions_annual",
            index_sets=zen_model.sets["set_time_steps_yearly"],
            doc="annual carbon emissions of energy system",
            unit_category={"emissions": 1},
        )
        # cumulative carbon emissions
        zen_model.variables.add_variable(
            name="carbon_emissions_cumulative",
            index_sets=zen_model.sets["set_time_steps_yearly"],
            doc="cumulative carbon emissions of energy system over time for each year",
            unit_category={"emissions": 1},
        )
        # carbon emission overshoot
        zen_model.variables.add_variable(
            name="carbon_emissions_budget_overshoot",
            index_sets=zen_model.sets["set_time_steps_yearly"],
            bounds=(0, np.inf),
            doc="overshoot carbon emissions of energy system "
            "at the end of the time horizon",
            unit_category={"emissions": 1},
        )
        # carbon emission overshoot
        zen_model.variables.add_variable(
            name="carbon_emissions_annual_overshoot",
            index_sets=zen_model.sets["set_time_steps_yearly"],
            bounds=(0, np.inf),
            doc="overshoot of the annual carbon emissions limit of energy system",
            unit_category={"emissions": 1},
        )
        # cost of carbon emissions
        zen_model.variables.add_variable(
            name="cost_carbon_emissions_total",
            index_sets=zen_model.sets["set_time_steps_yearly"],
            doc="total cost of carbon emissions of energy system",
            unit_category={"money": 1},
        )
        # costs
        zen_model.variables.add_variable(
            name="cost_total",
            index_sets=zen_model.sets["set_time_steps_yearly"],
            doc="total cost of energy system",
            unit_category={"money": 1},
        )
        # net_present_cost
        zen_model.variables.add_variable(
            name="net_present_cost",
            index_sets=zen_model.sets["set_time_steps_yearly"],
            doc="net_present_cost of energy system",
            unit_category={"money": 1},
        )

    def construct_constraints(self, zen_model: ZenModel):
        """Constructs the constraints of the class <EnergySystem>."""
        logger.info("Constructing constraints of EnergySystem")

        # create the rules
        self.rules = EnergySystemRules(self.config, self.context)
        # cumulative carbon emissions
        self.rules.constraint_carbon_emissions_cumulative(zen_model, self)

        # annual limit carbon emissions
        self.rules.constraint_carbon_emissions_annual_limit(zen_model, self)

        # carbon emission budget limit
        self.rules.constraint_carbon_emissions_budget(zen_model, self)

        # net_present_cost
        self.rules.constraint_net_present_cost(zen_model, self)

        # total carbon emissions
        self.rules.constraint_carbon_emissions_annual(zen_model, self)

        # cost of carbon emissions
        self.rules.constraint_cost_carbon_emissions_total(zen_model, self)

        # costs
        self.rules.constraint_cost_total(zen_model, self)

        # disable carbon emissions budget overshoot
        self.rules.constraint_carbon_emissions_budget_overshoot(zen_model, self)

        # disable annual carbon emissions overshoot
        self.rules.constraint_carbon_emissions_annual_overshoot(zen_model, self)

    def construct_objective(self, zen_model: ZenModel):
        """Constructs the pe.Objective of the class <EnergySystem>."""
        logger.info("Constructing objective for EnergySystem")

        # get selected objective rule
        if self.config.analysis.objective == "total_cost":
            objective = self.rules.objective_total_cost(zen_model)
        elif self.config.analysis.objective == "total_carbon_emissions":
            objective = self.rules.objective_total_carbon_emissions(zen_model)
        else:
            raise KeyError(f"Objective type {self.config.analysis.objective} not known")

        # get selected objective sense
        sense = self.config.analysis.sense
        assert sense in ["min", "max"], f"Objective sense {sense} not known"

        # construct objective
        zen_model.lp_model.add_objective(objective, sense=sense)

    def _add_parameter(
        self,
        zen_model: ZenModel,
        name: str,
        doc: str,
        index_names: list[str] | None = None,
        set_time_steps=None,
    ):
        """Adds a parameter to the zen_model.parameters.

        :param zen_model: The ZenModel of the EnergySystem class
        :param name: The name of the parameter
        :param index_names: The index names of the parameter
        """

        component_data, index_list, dict_of_units = self._initialize_component(
            zen_model, name, index_names, set_time_steps
        )
        component_data = self._ensure_pd_series_multi_index(component_data)
        data = component_data, index_list

        zen_model.parameters.add_parameter(
            name=name,
            doc=doc,
            data=data,
            dict_of_units=dict_of_units,
        )

    def _initialize_component(
        self,
        zen_model: ZenModel,
        component_name: str,
        index_names: list[str] | None,
        set_time_steps=None,
    ):
        """Initialize a modeling component by extracting the stored input data.

        Args:
            calling_class: class from where the method is called
            component_name: name of modeling component
            index_names: names of index sets, only if calling_class is not EnergySystem
            set_time_steps: time steps, only if calling_class is EnergySystem
            capacity_types: boolean if extracted for capacities
            component_data: data to initialize the component
        """
        component = getattr(self, component_name)
        dict_of_units = {}
        if component_name in self.units:
            dict_of_units = self.units[component_name]

        if index_names is not None:
            index_list = index_names
        elif set_time_steps is not None:
            index_list = [set_time_steps]
        else:
            index_list = []

        if set_time_steps:
            component_data = component[zen_model.sets[set_time_steps]]
        elif type(component) is float:
            component_data = component
        else:
            component_data = component.squeeze()

        return component_data, index_list, dict_of_units

    def _ensure_pd_series_multi_index(self, component_data):
        """Convert pd.Series index to pd.MultiIndex.

        :param component_data: extracted data as pd.Series
        :return: component_data: extracted data as pd.Series with MultiIndex
        """
        if isinstance(component_data, pd.Series) and not isinstance(
            component_data.index, pd.MultiIndex
        ):
            component_data.index = pd.MultiIndex.from_product(
                [component_data.index.to_list()]
            )
        return component_data


class EnergySystemRules(GenericRule):
    """This class takes care of the rules for the EnergySystem."""

    def __init__(self, config: Config, context: Context):
        """Inits the constraints for a given energy system.

        :param config: The Config of the entire setup
        :param context: The Context of the entire setup
        """
        super().__init__(config, context)

    def constraint_carbon_emissions_cumulative(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Cumulative carbon emissions over time.

        .. math::
            \\text{First planning period } y = y_0, \\quad E_y^\\mathrm{cum} = E_y
        .. math::
            \\text{Subsequent periods } y > y_0, \\quad E_y^{cum}
            = E_{y-1}^{cum} + (dy-1)E_{y-1}+E_y

        :math:`dy`: interval between planning periods \n
        :math:`E_y`: annual carbon emissions in year :math:`y` \n
        :math:`E_y^{cum}`: cumulative carbon emissions in year :math:`y`

        """
        m = [
            True if year == energy_system.set_time_steps_yearly[0] else False
            for year in energy_system.set_time_steps_yearly
        ]

        lhs = (
            zen_model.lp_model.variables["carbon_emissions_cumulative"]
            - zen_model.lp_model.variables["carbon_emissions_cumulative"].shift(
                set_time_steps_yearly=1
            )
            - zen_model.lp_model.variables["carbon_emissions_annual"].shift(
                set_time_steps_yearly=1
            )
            * (self.config.system.interval_between_years - 1)
            - zen_model.lp_model.variables["carbon_emissions_annual"]
        )
        rhs = (
            xr.ones_like(
                zen_model.lp_model.variables["carbon_emissions_cumulative"].mask
            )
            * zen_model.parameters.carbon_emissions_cumulative_existing
        ).where(m, 0)
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_cumulative", constraints
        )

    def constraint_carbon_emissions_annual_limit(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Time dependent carbon emissions limit from technologies and carriers.

        .. math::
            E_y\\leq e_y

        """
        lhs = (
            zen_model.lp_model.variables["carbon_emissions_annual"]
            - zen_model.lp_model.variables["carbon_emissions_annual_overshoot"]
        )
        rhs = zen_model.parameters.carbon_emissions_annual_limit
        constraints = lhs <= rhs

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_annual_limit", constraints
        )

    def constraint_carbon_emissions_budget(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Carbon emissions budget of whole time horizon.
        The prediction extends until the end of the horizon, i.e., last optimization
        time step plus the current carbon emissions until the end of the horizon.

        .. math::
            E_y^\\mathrm{cum} + (dy-1)  E_y - E_y^\\mathrm{bo} \\leq e^b

        :math:`E_y^\\mathrm{cum}`: cumulative carbon emissions of energy
        system in year :math:`y` \n
        :math:`E_y`: annual carbon emissions of energy system in year :math:`y` \n
        :math:`E_y^\\mathrm{bo}`: cumulative carbon emissions budget overshoot
        of energy system \n
        :math:`e^b`: carbon emissions budget of energy system

        """
        m = [
            year != energy_system.set_time_steps_yearly_entire_horizon[-1]
            for year in energy_system.set_time_steps_yearly
        ]

        lhs = (
            zen_model.lp_model.variables["carbon_emissions_cumulative"]
            - zen_model.lp_model.variables["carbon_emissions_budget_overshoot"]
            + (
                zen_model.lp_model.variables["carbon_emissions_annual"].where(m)
                * (self.config.system.interval_between_years - 1)
            )
        )
        rhs = zen_model.parameters.carbon_emissions_budget
        constraints = lhs <= rhs

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_budget", constraints
        )

    def constraint_net_present_cost(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Discounts the annual capital flows to calculate the net_present_cost.

        .. math::
            NPC_y = \\sum_{i \\in [0,dy(y))-1]}
            \\left( \\dfrac{1}{1+r} \\right)^{\\left(dy (y-y_0) + i \\right)} C_y

        :math:`NPC_y`: net present cost of energy system in year :math:`y` \n
        :math:`C_y`: total cost of energy system in year :math:`y` \n
        :math:`r`: discount rate \n
        :math:`dy`: interval between planning periods \n

        """
        factor = pd.Series(index=energy_system.set_time_steps_yearly)
        for year in energy_system.set_time_steps_yearly:
            ### auxiliary calculations
            if year == energy_system.set_time_steps_yearly_entire_horizon[-1]:
                interval_between_years = 1
            else:
                interval_between_years = self.config.system.interval_between_years
            # economic discount
            factor[year] = sum(
                (
                    (1 / (1 + zen_model.parameters.discount_rate))
                    ** (
                        self.config.system.interval_between_years
                        * (year - energy_system.set_time_steps_yearly[0])
                        + _intermediate_time_step
                    )
                )
                for _intermediate_time_step in range(0, interval_between_years)
            )
        term_discounted_cost_total = zen_model.lp_model.variables["cost_total"] * factor

        lhs = (
            zen_model.lp_model.variables["net_present_cost"]
            - term_discounted_cost_total
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint("constraint_net_present_cost", constraints)

    def constraint_carbon_emissions_budget_overshoot(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Enforces zero budget overshoot if price for budget overshoot is inf.

        ensures carbon emissions overshoot of carbon budget is zero when
        carbon emissions price for budget overshoot is inf.

        .. math::
            \\text{if } \\mu^{bo} =\\infty \\text{,then: }E_y^\\mathrm{bo} = 0

        :math:`E_y^\\mathrm{bo}`: overshoot carbon emissions of energy system at
        the end of the time horizon \n
        :math:`\\mu^{bo}`: carbon price for budget overshoot


        """
        if zen_model.parameters.price_carbon_emissions_budget_overshoot == np.inf:
            lhs = zen_model.lp_model.variables["carbon_emissions_budget_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = None

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_budget_overshoot", constraints
        )

    def constraint_carbon_emissions_annual_overshoot(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Enforces zero annual overshoot if price for annual overshoot is inf.

        ensures annual carbon emissions overshoot is zero when carbon
        emissions price for annual overshoot is inf.

        .. math::
            \\text{if } \\mu^o =\\infty \\text{,then: } E_y^\\mathrm{o} = 0

        :math:`E_y^\\mathrm{o}`: overshoot of the annual carbon emissions limit
        of energy system \n
        :math:`\\mu^o`: carbon price for annual overshoot

        """
        no_price = (
            zen_model.parameters.price_carbon_emissions_annual_overshoot == np.inf
        )
        no_limit = (zen_model.parameters.carbon_emissions_annual_limit == np.inf).all()
        if (no_price or no_limit) and not (no_price and no_limit):
            lhs = zen_model.lp_model.variables["carbon_emissions_annual_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = None

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_annual_overshoot", constraints
        )

    def constraint_carbon_emissions_annual(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Add up all carbon emissions from technologies and carriers.

        .. math::
            E_y = E_{y,\\mathcal{H}} + E_{y,\\mathcal{C}}

        :math:`E_{y,\\mathcal{H}}`: carbon emissions from technologies in
        year :math:`y` \n
        :math:`E_{y,\\mathcal{C}}`: carbon emissions from carriers in year :math

        """
        lhs = (
            zen_model.lp_model.variables["carbon_emissions_annual"]
            - zen_model.lp_model.variables["carbon_emissions_technology_total"]
            - zen_model.lp_model.variables["carbon_emissions_carrier_total"]
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_annual", constraints
        )

    def constraint_cost_carbon_emissions_total(
        self, zen_model: ZenModel, energy_system: EnergySystem
    ):
        """Carbon cost associated with the carbon emissions of the system in each year.

        .. math::
            OPEX_y^\\mathrm{c} = E_y\\mu + E_y^\\mathrm{o}\\mu^\\mathrm{o}

        :math:`OPEX_y^\\mathrm{c}`: cost of carbon emissions in year :math:`y` \n
        :math:`E_y`: annual carbon emissions of energy system in year :math:`y` \n
        :math:`\\mu`: carbon price \n
        :math:`E_y^\\mathrm{o}`: annual carbon emissions overshoot in year :math:`y` \n
        :math:`\\mu^\\mathrm{o}`: carbon price for annual overshoot

        """
        mask_last_year = [
            year == energy_system.set_time_steps_yearly[-1]
            for year in energy_system.set_time_steps_yearly
        ]

        lhs = (
            zen_model.lp_model.variables["cost_carbon_emissions_total"]
            - zen_model.lp_model.variables["carbon_emissions_annual"]
            * zen_model.parameters.price_carbon_emissions
        )
        # add cost for overshooting carbon emissions budget
        if zen_model.parameters.price_carbon_emissions_budget_overshoot != np.inf:
            lhs -= (
                zen_model.lp_model.variables["carbon_emissions_budget_overshoot"].where(
                    mask_last_year
                )
                * zen_model.parameters.price_carbon_emissions_budget_overshoot.item()
            )
        # add cost for overshooting annual carbon emissions limit
        if zen_model.parameters.price_carbon_emissions_annual_overshoot != np.inf:
            lhs -= (
                zen_model.lp_model.variables["carbon_emissions_annual_overshoot"]
                * zen_model.parameters.price_carbon_emissions_annual_overshoot.item()
            )

        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint(
            "constraint_cost_carbon_emissions_total", constraints
        )

    def constraint_cost_total(self, zen_model: ZenModel, energy_system: EnergySystem):
        """Add up all costs from technologies and carriers.

        .. math::
            C_y = CAPEX_y + OPEX_y^\\mathrm{t} + OPEX_y^\\mathrm{c} + OPEX_y^\\mathrm{e}

        :math:`C_y`: total cost of energy system in year :math:`y` \n
        :math:`CAPEX_y`: annual capital expenditures in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{t}`: annual operational expenditures for operating
        technologies in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{c}`: annual operational expenditures for for importing
        and exporting carriers in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{e}`: annual operational expenditures for carbon
        emissions in year :math:`y`

        """
        lhs = (
            zen_model.lp_model.variables["cost_total"]
            - zen_model.lp_model.variables["cost_capex_yearly_total"]
            - zen_model.lp_model.variables["cost_opex_yearly_total"]
            - zen_model.lp_model.variables["cost_carrier_total"]
            - zen_model.lp_model.variables["cost_carbon_emissions_total"]
        )
        rhs = 0
        constraints = lhs == rhs

        zen_model.constraints.add_constraint("constraint_cost_total", constraints)

    # Objective rules
    # ---------------

    def objective_total_cost(self, zen_model: ZenModel):
        """Objective function to minimize the total net present cost.

        .. math::
            J = \\sum_{y\\in\\mathcal{Y}} NPC_y

        :param model: optimization model
        :return: net present cost objective function
        """
        return zen_model.lp_model.variables["net_present_cost"].sum(
            "set_time_steps_yearly"
        )

    def objective_total_carbon_emissions(self, zen_model: ZenModel):
        """Objective function to minimize total emissions.

        .. math::
            J = E^{\\mathrm{cum}}_Y

        :math:`E^{\\mathrm{cum}}_Y`: cumulative carbon emissions at the end of
        the time horizon

        :param model: optimization model
        :return: total carbon emissions objective function
        """
        sets = zen_model.sets
        return (
            zen_model.lp_model.variables["carbon_emissions_cumulative"]
            .at[sets["set_time_steps_yearly"][-1]]
            .to_linexpr()
        )
