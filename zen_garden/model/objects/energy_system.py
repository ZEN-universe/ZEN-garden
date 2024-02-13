"""
:Title:          ZEN-GARDEN
:Created:        January-2022
:Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Class defining a standard EnergySystem. Contains methods to add parameters, variables and constraints to the
optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
optimization model as an input.
"""
import copy
import logging

import numpy as np

from zen_garden.model.objects.element import GenericRule,Element
from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.preprocess.unit_handling import UnitHandling
from .time_steps import TimeStepsDicts
from pathlib import Path

class EnergySystem:
    """
    Class defining a standard energy system
    """
    def __init__(self, optimization_setup):
        """ initialization of the energy_system

        :param optimization_setup: The OptimizationSetup of the EnergySystem class"""

        # the name
        self.name = "EnergySystem"
        self._name = "EnergySystem"
        # set attributes
        self.optimization_setup = optimization_setup
        # quick access
        self.system = self.optimization_setup.system
        # empty dict of technologies of carrier
        self.dict_technology_of_carrier = {}
        # The timesteps
        self.time_steps = TimeStepsDicts()

        # empty list of indexing sets
        self.indexing_sets = []

        # set indexing sets
        for key in self.system:
            if "set" in key:
                self.indexing_sets.append(key)

        # set input path
        folder_label = "energy_system"
        self.input_path = Path(self.optimization_setup.paths[folder_label]["folder"])

        # create UnitHandling object
        self.unit_handling = UnitHandling(self.input_path,
                                          self.optimization_setup.solver["rounding_decimal_points_units"],
                                          self.optimization_setup.solver["define_ton_as_metric_ton"])

        # create DataInput object
        self.data_input = DataInput(element=self, system=self.system,
                                    analysis=self.optimization_setup.analysis, solver=self.optimization_setup.solver,
                                    energy_system=self, unit_handling=self.unit_handling)
        # initialize empty set_carriers list
        self.set_carriers = []
        #dict to save the parameter units (and save them in the results later on)
        self.units = {}

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # store scenario dict
        self.data_input.scenario_dict = self.optimization_setup.scenario_dict
        # in class <EnergySystem>, all sets are constructed
        self.set_nodes = self.data_input.extract_locations()
        self.set_nodes_on_edges = self.calculate_edges_from_nodes()
        self.set_edges = list(self.set_nodes_on_edges.keys())
        self.set_haversine_distances_edges = self.calculate_haversine_distances_from_nodes()
        self.set_technologies = self.system["set_technologies"]
        # base time steps
        self.set_base_time_steps = list(range(0, self.system["unaggregated_time_steps_per_year"] * self.system["optimized_years"]))
        self.set_base_time_steps_yearly = list(range(0, self.system["unaggregated_time_steps_per_year"]))

        # yearly time steps
        self.set_time_steps_yearly = list(range(self.system["optimized_years"]))
        self.set_time_steps_yearly_entire_horizon = copy.deepcopy(self.set_time_steps_yearly)
        time_steps_yearly_duration = self.time_steps.calculate_time_step_duration(self.set_time_steps_yearly, self.set_base_time_steps)
        self.sequence_time_steps_yearly = np.concatenate([[time_step] * time_steps_yearly_duration[time_step] for time_step in time_steps_yearly_duration])
        self.time_steps.sequence_time_steps_yearly = self.sequence_time_steps_yearly
        # list containing simulated years (needed for convert_real_to_generic_time_indices() in extract_input_data.py)
        self.set_time_steps_years = list(range(self.system["reference_year"],self.system["reference_year"] + self.system["optimized_years"]*self.system["interval_between_years"],self.system["interval_between_years"]))
        # parameters whose time-dependant data should not be interpolated (for years without data) in the extract_input_data.py convertRealToGenericTimeIndices() function
        self.parameters_interpolation_off = self.data_input.read_input_csv("parameters_interpolation_off")
        # technology-specific
        self.set_conversion_technologies = self.system["set_conversion_technologies"]
        self.set_transport_technologies = self.system["set_transport_technologies"]
        self.set_storage_technologies = self.system["set_storage_technologies"]
        self.set_retrofitting_technologies= self.system["set_retrofitting_technologies"]
        # discount rate
        self.discount_rate = self.data_input.extract_input_data("discount_rate", index_sets=[], unit_category={})
        # carbon emissions limit
        self.carbon_emissions_annual_limit = self.data_input.extract_input_data("carbon_emissions_annual_limit", index_sets=["set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"emissions": 1})
        _fraction_year = self.system["unaggregated_time_steps_per_year"] / self.system["total_hours_per_year"]
        self.carbon_emissions_annual_limit = self.carbon_emissions_annual_limit * _fraction_year  # reduce to fraction of year
        self.carbon_emissions_budget = self.data_input.extract_input_data("carbon_emissions_budget", index_sets=[], unit_category={"emissions": 1})
        self.carbon_emissions_cumulative_existing = self.data_input.extract_input_data("carbon_emissions_cumulative_existing", index_sets=[], unit_category={"emissions": 1})
        # price carbon emissions
        self.price_carbon_emissions = self.data_input.extract_input_data("price_carbon_emissions", index_sets=["set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"money": 1, "emissions": -1})
        self.price_carbon_emissions_budget_overshoot = self.data_input.extract_input_data("price_carbon_emissions_budget_overshoot", index_sets=[], unit_category={"money": 1, "emissions": -1})
        self.price_carbon_emissions_annual_overshoot = self.data_input.extract_input_data("price_carbon_emissions_annual_overshoot", index_sets=[], unit_category={"money": 1, "emissions": -1})
        # market share unbounded
        self.market_share_unbounded = self.data_input.extract_input_data("market_share_unbounded", index_sets=[], unit_category={})
        # knowledge_spillover_rate
        self.knowledge_spillover_rate = self.data_input.extract_input_data("knowledge_spillover_rate", index_sets=[], unit_category={})

    def calculate_edges_from_nodes(self):
        """ calculates set_nodes_on_edges from set_nodes

        :return set_nodes_on_edges: dict with edges and corresponding nodes """

        set_nodes_on_edges = {}
        # read edge file
        set_edges_input = self.data_input.extract_locations(extract_nodes=False)
        for edge in set_edges_input.index:
            set_nodes_on_edges[edge] = (set_edges_input.loc[edge, "node_from"], set_edges_input.loc[edge, "node_to"])
        return set_nodes_on_edges

    def calculate_haversine_distances_from_nodes(self):
        """
        Computes the distance in kilometers between two nodes by using their lon lat coordinates and the Haversine formula

        :return: dict containing all edges along with their distances
        """
        set_haversine_distances_of_edges = {}
        # read coords file
        df_coords_input = self.data_input.extract_locations(extract_coordinates=True)
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
            a = np.sin(dlat / 2) ** 2 + np.cos(coords1["lat"].squeeze()) * np.cos(coords2["lat"].squeeze()) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = radius * c
            set_haversine_distances_of_edges[edge] = distance
        multiplier = self.unit_handling.get_unit_multiplier("km", attribute_name="distance")
        set_haversine_distances_of_edges = {key: value * multiplier for key, value in set_haversine_distances_of_edges.items()}
        return set_haversine_distances_of_edges

    def set_technology_of_carrier(self, technology, list_technology_of_carrier):
        """ appends technology to carrier in dict_technology_of_carrier

        :param technology: name of technology in model
        :param list_technology_of_carrier: list of carriers correspondent to technology"""
        for carrier in list_technology_of_carrier:
            if carrier not in self.dict_technology_of_carrier:
                self.dict_technology_of_carrier[carrier] = [technology]
                self.set_carriers.append(carrier)
            elif technology not in self.dict_technology_of_carrier[carrier]:
                self.dict_technology_of_carrier[carrier].append(technology)

    def calculate_connected_edges(self, node, direction: str):
        """ calculates connected edges going in (direction = 'in') or going out (direction = 'out')

        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode, out: node = startnode
        :return _set_connected_edges: list of connected edges """
        if direction == "in":
            # second entry is node into which the flow goes
            _set_connected_edges = [edge for edge in self.set_nodes_on_edges if self.set_nodes_on_edges[edge][1] == node]
        elif direction == "out":
            # first entry is node out of which the flow starts
            _set_connected_edges = [edge for edge in self.set_nodes_on_edges if self.set_nodes_on_edges[edge][0] == node]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return _set_connected_edges

    def calculate_reversed_edge(self, edge):
        """ calculates the reversed edge corresponding to an edge

        :param edge: input edge
        :return _reversed_edge: edge which corresponds to the reversed direction of edge"""
        _node_out, _node_in = self.set_nodes_on_edges[edge]
        for _reversed_edge in self.set_nodes_on_edges:
            if _node_out == self.set_nodes_on_edges[_reversed_edge][1] and _node_in == self.set_nodes_on_edges[_reversed_edge][0]:
                return _reversed_edge
        raise KeyError(f"Edge {edge} has no reversed edge. However, at least one transport technology is bidirectional")

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to EnergySystem --- ###

    def construct_sets(self):
        """ constructs the pe.Sets of the class <EnergySystem> """
        # construct pe.Sets of the class <EnergySystem>
        # nodes
        self.optimization_setup.sets.add_set(name="set_nodes", data=self.set_nodes, doc="Set of nodes")
        # edges
        self.optimization_setup.sets.add_set(name="set_edges", data=self.set_edges, doc="Set of edges")
        # nodes on edges
        self.optimization_setup.sets.add_set(name="set_nodes_on_edges", data=self.set_nodes_on_edges, doc="Set of nodes that constitute an edge. Edge connects first node with second node.",
                                             index_set="set_edges")
        # carriers
        self.optimization_setup.sets.add_set(name="set_carriers", data=self.set_carriers, doc="Set of carriers")
        # technologies
        self.optimization_setup.sets.add_set(name="set_technologies", data=self.set_technologies, doc="set_technologies")
        # all elements
        data = list(set(self.optimization_setup.sets["set_technologies"]) | set(self.optimization_setup.sets["set_carriers"]))
        self.optimization_setup.sets.add_set(name="set_elements", data=data, doc="Set of elements")
        # set set_elements to indexing_sets
        self.indexing_sets.append("set_elements")
        # time-steps
        self.optimization_setup.sets.add_set(name="set_base_time_steps", data=self.set_base_time_steps, doc="Set of base time-steps")
        # yearly time steps
        self.optimization_setup.sets.add_set(name="set_time_steps_yearly", data=self.set_time_steps_yearly, doc="Set of yearly time-steps")
        # yearly time steps of entire optimization horizon
        self.optimization_setup.sets.add_set(name="set_time_steps_yearly_entire_horizon", data=self.set_time_steps_yearly_entire_horizon, doc="Set of yearly time-steps of entire optimization horizon")
        # operational time steps
        self.optimization_setup.sets.add_set(name="set_time_steps_operation",data=self.time_steps.time_steps_operation,doc="Set of operational time steps")
        # storage time steps
        self.optimization_setup.sets.add_set(name="set_time_steps_storage",data=self.time_steps.time_steps_storage,doc="Set of storage level time steps")

    def construct_params(self):
        """ constructs the pe.Params of the class <EnergySystem> """

        cls = self.__class__
        parameters = self.optimization_setup.parameters
        # operational time step duration
        parameters.add_parameter(name="time_steps_operation_duration",
                                 data=self.optimization_setup.initialize_component(cls, "time_steps_operation_duration", set_time_steps="set_time_steps_operation"),
                                 doc="Parameter which specifies the duration of each operational time step")
        # storage time step duration
        parameters.add_parameter(name="time_steps_storage_duration",
                                 data=self.optimization_setup.initialize_component(cls, "time_steps_storage_duration", set_time_steps="set_time_steps_storage"),
                                 doc="Parameter which specifies the duration of each storage time step")
        # discount rate
        parameters.add_parameter(name="discount_rate",
             data=self.optimization_setup.initialize_component(cls, "discount_rate"),
             doc='Parameter which specifies the discount rate of the energy system')
        # carbon emissions limit
        parameters.add_parameter(name="carbon_emissions_annual_limit", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_annual_limit", set_time_steps="set_time_steps_yearly"),
            doc='Parameter which specifies the total limit on carbon emissions')
        # carbon emissions budget
        parameters.add_parameter(name="carbon_emissions_budget", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_budget"),
            doc='Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon')
        # carbon emissions budget
        parameters.add_parameter(name="carbon_emissions_cumulative_existing", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_cumulative_existing"), doc='Parameter which specifies the total previous carbon emissions')
        # carbon price
        parameters.add_parameter(name="price_carbon_emissions", data=self.optimization_setup.initialize_component(cls, "price_carbon_emissions", set_time_steps="set_time_steps_yearly"),
            doc='Parameter which specifies the yearly carbon price')
        # carbon price of budget overshoot
        parameters.add_parameter(name="price_carbon_emissions_budget_overshoot", data=self.optimization_setup.initialize_component(cls,"price_carbon_emissions_budget_overshoot"),
                                 doc='Parameter which specifies the carbon price for budget overshoot')
        # carbon price of annual overshoot
        parameters.add_parameter(name="price_carbon_emissions_annual_overshoot", data=self.optimization_setup.initialize_component(cls, "price_carbon_emissions_annual_overshoot"),
                                 doc='Parameter which specifies the carbon price for annual overshoot')
        # carbon price of overshoot
        parameters.add_parameter(name="market_share_unbounded", data=self.optimization_setup.initialize_component(cls, "market_share_unbounded"),
                                 doc='Parameter which specifies the unbounded market share')
        # carbon price of overshoot
        parameters.add_parameter(name="knowledge_spillover_rate", data=self.optimization_setup.initialize_component(cls, "knowledge_spillover_rate"),
                                 doc='Parameter which specifies the knowledge spillover rate')

    def construct_vars(self):
        """ constructs the pe.Vars of the class <EnergySystem> """
        variables = self.optimization_setup.variables
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model
        # carbon emissions
        variables.add_variable(model, name="carbon_emissions_annual", index_sets=sets["set_time_steps_yearly"], doc="annual carbon emissions of energy system")
        # cumulative carbon emissions
        variables.add_variable(model, name="carbon_emissions_cumulative", index_sets=sets["set_time_steps_yearly"],
                               doc="cumulative carbon emissions of energy system over time for each year")
        # carbon emission overshoot
        variables.add_variable(model, name="carbon_emissions_budget_overshoot", index_sets=sets["set_time_steps_yearly"], bounds=(0, np.inf),
                               doc="overshoot carbon emissions of energy system at the end of the time horizon")
        # carbon emission overshoot
        variables.add_variable(model, name="carbon_emissions_annual_overshoot", index_sets=sets["set_time_steps_yearly"], bounds=(0, np.inf),
                               doc="overshoot of the annual carbon emissions limit of energy system")
        # cost of carbon emissions
        variables.add_variable(model, name="cost_carbon_emissions_total", index_sets=sets["set_time_steps_yearly"],
                               doc="total cost of carbon emissions of energy system")
        # costs
        variables.add_variable(model, name="cost_total", index_sets=sets["set_time_steps_yearly"],
                               doc="total cost of energy system")
        # net_present_cost
        variables.add_variable(model, name="net_present_cost", index_sets=sets["set_time_steps_yearly"],
                               doc="net_present_cost of energy system")

    def construct_constraints(self):
        """ constructs the pe.Constraints of the class <EnergySystem> """

        constraints = self.optimization_setup.constraints
        sets = self.optimization_setup.sets
        model = self.optimization_setup.model

        # create the rules
        self.rules = EnergySystemRules(self.optimization_setup)
        # cumulative carbon emissions
        constraints.add_constraint_rule(model, name="constraint_carbon_emissions_cumulative", index_sets=sets["set_time_steps_yearly"], rule=self.rules.constraint_carbon_emissions_cumulative_rule,
                                        doc="cumulative carbon emissions of energy system over time")
        # annual limit carbon emissions
        constraints.add_constraint_rule(model, name="constraint_carbon_emissions_annual_limit", index_sets=sets["set_time_steps_yearly"], rule=self.rules.constraint_carbon_emissions_annual_limit_rule,
                                   doc="limit of total annual carbon emissions of energy system")
        # carbon emission budget limit
        constraints.add_constraint_rule(model, name="constraint_carbon_emissions_budget", index_sets=sets["set_time_steps_yearly"], rule=self.rules.constraint_carbon_emissions_budget_rule,
                                   doc="Budget of total carbon emissions of energy system")
        # net_present_cost
        constraints.add_constraint_rule(model, name="constraint_net_present_cost", index_sets=sets["set_time_steps_yearly"], rule=self.rules.constraint_net_present_cost_rule, doc="net_present_cost of energy system")
        # total carbon emissions
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_annual", constraint=self.rules.constraint_carbon_emissions_annual_block(),
                                         doc="total annual carbon emissions of energy system")
        # cost of carbon emissions
        constraints.add_constraint_block(model, name="constraint_cost_carbon_emissions_total", constraint=self.rules.constraint_cost_carbon_emissions_total_block(),
                                         doc="total carbon emissions cost of energy system")
        # costs
        constraints.add_constraint_block(model, name="constraint_cost_total", constraint=self.rules.constraint_cost_total_block(),
                                         doc="total cost of energy system")
        # disable carbon emissions budget overshoot
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_budget_overshoot", constraint=self.rules.constraint_carbon_emissions_budget_overshoot_block(),
                                        doc="disable carbon emissions budget overshoot if carbon emissions budget overshoot price is inf")
        # disable annual carbon emissions overshoot
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_annual_overshoot", constraint=self.rules.constraint_carbon_emissions_annual_overshoot_block(),
                                        doc="disable annual carbon emissions overshoot if annual carbon emissions overshoot price is inf")

    def construct_objective(self):
        """ constructs the pe.Objective of the class <EnergySystem> """
        logging.info("Construct pe.Objective")

        # get selected objective rule
        if self.optimization_setup.analysis["objective"] == "total_cost":
            objective_rule = self.rules.objective_total_cost_rule(self.optimization_setup.model)
        elif self.optimization_setup.analysis["objective"] == "total_carbon_emissions":
            objective_rule = self.rules.objective_total_carbon_emissions_rule(self.optimization_setup.model)
        elif self.optimization_setup.analysis["objective"] == "risk":
            logging.info("Objective of minimizing risk not yet implemented")
            objective_rule = self.rules.objective_risk_rule(self.optimization_setup.model)
        else:
            raise KeyError(f"Objective type {self.optimization_setup.analysis['objective']} not known")

        # get selected objective sense
        if self.optimization_setup.analysis["sense"] == "minimize":
            logging.info("Using sense 'minimize'")
        elif self.optimization_setup.analysis["sense"] == "maximize":
            raise NotImplementedError("Currently only minimization supported")
        else:
            raise KeyError(f"Objective sense {self.optimization_setup.analysis['sense']} not known")

        # construct objective
        self.optimization_setup.model.add_objective(objective_rule.to_linexpr())


class EnergySystemRules(GenericRule):
    """
    This class takes care of the rules for the EnergySystem
    """

    def __init__(self, optimization_setup):
        """
        Inits the constraints for a given energy system

        :param optimization_setup: The OptimizationSetup of the EnergySystem class
        """

        super().__init__(optimization_setup)

    # Rule-based constraints
    # ----------------------

    def constraint_carbon_emissions_cumulative_rule(self, year):
        """ cumulative carbon emissions over time

        .. math::
            \mathrm{First\ planning\ period}\ y = y_0,\quad E_y^\mathrm{c} = E_y
        .. math::
            \mathrm{Subsequent\ periods}\ y > y_0, \quad E_y^c = E_{y-1}^c + (\Delta^y-1)E_{y-1}+E_y

        :param year: year of interest
        :return: cumulative carbon emissions constraint for specified year
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        if year == self.optimization_setup.sets["set_time_steps_yearly"][0]:
            lhs = self.variables["carbon_emissions_cumulative"][year] - self.variables["carbon_emissions_annual"][year]
            rhs = self.parameters.carbon_emissions_cumulative_existing
            constraints = lhs == rhs
        else:
            lhs = (self.variables["carbon_emissions_cumulative"][year]
                   - self.variables["carbon_emissions_cumulative"][year - 1]
                   - self.variables["carbon_emissions_annual"][year - 1] * (self.system["interval_between_years"] - 1)
                   - self.variables["carbon_emissions_annual"][year])
            rhs = 0
            constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_carbon_emissions_annual_limit_rule(self, year):
        """ time dependent carbon emissions limit from technologies and carriers

        .. math::
            E_y\leq e_y

        :param year: year of interest
        :return: carbon emissions limit constraint for specified year
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = self.variables["carbon_emissions_annual"][year] - self.variables["carbon_emissions_annual_overshoot"][year]
        rhs = self.parameters.carbon_emissions_annual_limit.loc[year].item()
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_carbon_emissions_budget_rule(self, year):
        """ carbon emissions budget of entire time horizon from technologies and carriers.
        The prediction extends until the end of the horizon, i.e.,
        last optimization time step plus the current carbon emissions until the end of the horizon

        #TODO constraint doesn't match model formulation definition

        :param year: year of interest
        :return: carbon emissions budget constraint for specified year"""

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        if self.parameters.carbon_emissions_budget != np.inf:
            if year == self.optimization_setup.sets["set_time_steps_yearly_entire_horizon"][-1]:
                lhs = self.variables["carbon_emissions_cumulative"][year] - self.variables["carbon_emissions_budget_overshoot"][year]
                rhs = self.parameters.carbon_emissions_budget
                constraints = lhs <= rhs
            else:
                lhs = (self.variables["carbon_emissions_cumulative"][year] - self.variables["carbon_emissions_budget_overshoot"][year]
                       + self.variables["carbon_emissions_annual"][year] * (self.system["interval_between_years"] - 1))
                rhs = self.parameters.carbon_emissions_budget
                constraints = lhs <= rhs
        else:
            constraints = None

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_net_present_cost_rule(self, year):
        """ discounts the annual capital flows to calculate the net_present_cost

        .. math::
            NPC_y = C_y \sum_{\\tilde{y} = 1}^{\Delta^\mathrm{y}-1}(\\frac{1}{1+r})^{\Delta^\mathrm{y}(y-y_0)+\\tilde{y}}

        :param year: year of interest
        :return: net present cost constraint for specified year
       """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        if year == self.sets["set_time_steps_yearly_entire_horizon"][-1]:
            interval_between_years = 1
        else:
            interval_between_years = self.system["interval_between_years"]
        # economic discount
        factor = sum(((1 / (1 + self.parameters.discount_rate)) ** (self.system["interval_between_years"] * (year - self.sets["set_time_steps_yearly"][0]) + _intermediate_time_step))
                     for _intermediate_time_step in range(0, interval_between_years))
        term_discounted_cost_total = self.variables["cost_total"][year] * factor

        ### formulate constraint
        lhs = self.variables["net_present_cost"][year] - term_discounted_cost_total
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    # Block-based constraints
    # -----------------------

    def constraint_carbon_emissions_budget_overshoot_block(self):
        """ ensures carbon emissions overshoot of carbon budget is zero when carbon emissions price for budget overshoot is inf

        .. math::
            E_y^\mathrm{o} = 0

        :return: carbon emissions budget overshoot
        """

        ### index sets
        # not necessary

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        if self.parameters.price_carbon_emissions_budget_overshoot == np.inf:
            lhs = self.variables["carbon_emissions_budget_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = []

        return self.constraints.return_contraints(constraints)

    def constraint_carbon_emissions_annual_overshoot_block(self):
        """ ensures annual carbon emissions overshoot is zero when carbon emissions price for annual overshoot is inf

        .. math::
            E_y^\mathrm{o}

        :return: annual carbon emissions overshoot
        """

        ### index sets
        # not necessary

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        if self.parameters.price_carbon_emissions_annual_overshoot == np.inf or self.parameters.carbon_emissions_annual_limit.sum() == np.inf:
            lhs = self.variables["carbon_emissions_annual_overshoot"]
            rhs = 0
            constraints = lhs == rhs
        else:
            constraints = []

        return self.constraints.return_contraints(constraints)


    def constraint_carbon_emissions_annual_block(self):
        """ add up all carbon emissions from technologies and carriers

        .. math::
            E_y = E_{y,\mathcal{H}} + E_{y,\mathcal{C}}

        :return: total carbon emissions constraint for specified year
        """

        ### index sets
        # not necessary

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = (self.variables["carbon_emissions_annual"]
               - self.variables["carbon_emissions_technology_total"]
               - self.variables["carbon_emissions_carrier_total"])
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_cost_carbon_emissions_total_block(self):
        """ carbon cost associated with the carbon emissions of the system in each year

        .. math::
            OPEX_y^\mathrm{c} = E_y\mu + E_y^\mathrm{o}\mu^\mathrm{o}

        :return: total cost carbon emissions constraint for specified year
        """

        ### index sets
        # not necessary

        ### masks
        mask_last_year = [year == self.sets["set_time_steps_yearly"][-1] for year in self.sets["set_time_steps_yearly"]]

        ### index loop
        # not necessary

        ### auxiliary calculations
        lhs = (self.variables["cost_carbon_emissions_total"]
                   - self.variables["carbon_emissions_annual"] * self.parameters.price_carbon_emissions)
        # add cost for overshooting carbon emissions budget
        if self.parameters.price_carbon_emissions_budget_overshoot != np.inf:
            lhs -= self.variables["carbon_emissions_budget_overshoot"].where(mask_last_year) * self.parameters.price_carbon_emissions_budget_overshoot
        # add cost for overshooting annual carbon emissions limit
        if self.parameters.price_carbon_emissions_annual_overshoot != np.inf:
            lhs -= self.variables["carbon_emissions_annual_overshoot"] * self.parameters.price_carbon_emissions_annual_overshoot

        ### formulate constraint
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_cost_total_block(self):
        """ add up all costs from technologies and carriers

        .. math::
            OPEX_y^\mathrm{c} = E_y\mu + E_y^\mathrm{o}\mu^\mathrm{o}

        :return: total cost carbon emissions constraint for specified year
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = (self.variables["cost_total"]
               - self.variables["cost_capex_total"]
               - self.variables["cost_opex_total"]
               - self.variables["cost_carrier_total"]
               - self.variables["cost_carbon_emissions_total"])
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    # Objective rules
    # ---------------

    def objective_total_cost_rule(self, model):
        """objective function to minimize the total net present cost

        .. math::
            J = \sum_{y\in\mathcal{Y}} NPC_y

        :param model: optimization model
        :return: net present cost objective function
        """
        sets = self.sets
        return sum(model.variables["net_present_cost"][year] for year in sets["set_time_steps_yearly"])

    def objective_total_carbon_emissions_rule(self, model):
        """objective function to minimize total emissions

        .. math::
            J = \sum_{y\in\mathcal{Y}} E_y

        :param model: optimization model
        :return: total carbon emissions objective function
        """
        sets = self.sets
        return sum(model.variables["carbon_emissions_annual"][year] for year in sets["set_time_steps_yearly"])

    def objective_risk_rule(self, model):
        """objective function to minimize total risk

        #TODO add latex formula as soon as risk objective is implemented

        :param model: optimization model
        :return: risk objective function
        """
        # TODO implement objective functions for risk
        return None
