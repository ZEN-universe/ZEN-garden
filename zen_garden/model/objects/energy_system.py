"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        January-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard EnergySystem. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import copy
import logging

import numpy as np
import pyomo.environ as pe

from zen_garden.preprocess.functions.extract_input_data import DataInput
from zen_garden.preprocess.functions.unit_handling import UnitHandling
from .time_steps import TimeStepsDicts


class EnergySystem:

    def __init__(self, optimization_setup):
        """ initialization of the energy_system
        :param optimization_setup: The OptimizationSetup of the EnergySystem class"""

        # the name
        self.name = "energy_system"
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
        _folder_label = self.optimization_setup.analysis["folder_name_system_specification"]
        self.input_path = self.optimization_setup.paths[_folder_label]["folder"]

        # create UnitHandling object
        self.unit_handling = UnitHandling(self.input_path, self.optimization_setup.solver["rounding_decimal_points"])

        # create DataInput object
        self.data_input = DataInput(element=self, system=self.system,
                                    analysis=self.optimization_setup.analysis, solver=self.optimization_setup.solver,
                                    energy_system=self, unit_handling=self.unit_handling)

        # store input data
        self.store_input_data()

        # create the rules
        self.rules = EnergySystemRules(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """

        # in class <EnergySystem>, all sets are constructed
        self.set_nodes = self.data_input.extract_locations()
        self.set_nodes_on_edges = self.calculate_edges_from_nodes()
        self.set_edges = list(self.set_nodes_on_edges.keys())
        self.set_carriers = []
        self.set_technologies = self.system["set_technologies"]
        # base time steps
        self.set_base_time_steps = list(range(0, self.system["unaggregated_time_steps_per_year"] * self.system["optimized_years"]))
        self.set_base_time_steps_yearly = list(range(0, self.system["unaggregated_time_steps_per_year"]))

        # yearly time steps
        self.set_time_steps_yearly = list(range(self.system["optimized_years"]))
        self.set_time_steps_yearly_entire_horizon = copy.deepcopy(self.set_time_steps_yearly)
        time_steps_yearly_duration = self.time_steps.calculate_time_step_duration(self.set_time_steps_yearly, self.set_base_time_steps)
        self.sequence_time_steps_yearly = np.concatenate([[time_step] * time_steps_yearly_duration[time_step] for time_step in time_steps_yearly_duration])
        self.time_steps.set_sequence_time_steps(None, self.sequence_time_steps_yearly, time_step_type="yearly")
        # list containing simulated years (needed for convert_real_to_generic_time_indices() in extract_input_data.py)
        self.set_time_step_years = list(range(self.system["reference_year"],self.system["reference_year"] + self.system["optimized_years"]*self.system["interval_between_years"],self.system["interval_between_years"]))
        # parameters whose time-dependant data should not be interpolated (for years without data) in the extract_input_data.py convertRealToGenericTimeIndices() function
        self.parameters_interpolation_off = self.data_input.read_input_data("parameters_interpolation_off")
        # technology-specific
        self.set_conversion_technologies = self.system["set_conversion_technologies"]
        self.set_transport_technologies = self.system["set_transport_technologies"]
        self.set_storage_technologies = self.system["set_storage_technologies"]
        # carbon emissions limit
        self.carbon_emissions_limit = self.data_input.extract_input_data("carbon_emissions_limit", index_sets=["set_time_steps_yearly"], time_steps=self.set_time_steps_yearly)
        _fraction_year = self.system["unaggregated_time_steps_per_year"] / self.system["total_hours_per_year"]
        self.carbon_emissions_limit = self.carbon_emissions_limit * _fraction_year  # reduce to fraction of year
        self.carbon_emissions_budget = self.data_input.extract_input_data("carbon_emissions_budget", index_sets=[])
        self.previous_carbon_emissions = self.data_input.extract_input_data("previous_carbon_emissions", index_sets=[])
        # carbon price
        self.carbon_price = self.data_input.extract_input_data("carbon_price", index_sets=["set_time_steps_yearly"], time_steps=self.set_time_steps_yearly)
        self.carbon_price_overshoot = self.data_input.extract_input_data("carbon_price_overshoot", index_sets=[])

    def calculate_edges_from_nodes(self):
        """ calculates set_nodes_on_edges from set_nodes
        :return set_nodes_on_edges: dict with edges and corresponding nodes """

        set_nodes_on_edges = {}
        # read edge file
        set_edges_input = self.data_input.extract_locations(extract_nodes=False)
        if set_edges_input is not None:
            for edge in set_edges_input.index:
                set_nodes_on_edges[edge] = (set_edges_input.loc[edge, "node_from"], set_edges_input.loc[edge, "node_to"])
        else:
            logging.warning(f"DeprecationWarning: Implicit creation of edges will be deprecated. Provide 'set_edges.csv' in folder '{self.system['''folder_name_system_specification''']}' instead!")
            for node_from in self.set_nodes:
                for node_to in self.set_nodes:
                    if node_from != node_to:
                        set_nodes_on_edges[node_from + "-" + node_to] = (node_from, node_to)
        return set_nodes_on_edges

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
        pyomo_model = self.optimization_setup.model
        # nodes
        pyomo_model.set_nodes = pe.Set(initialize=self.set_nodes, doc='Set of nodes')
        # edges
        pyomo_model.set_edges = pe.Set(initialize=self.set_edges, doc='Set of edges')
        # nodes on edges
        pyomo_model.set_nodes_on_edges = pe.Set(pyomo_model.set_edges, initialize=self.set_nodes_on_edges, doc='Set of nodes that constitute an edge. Edge connects first node with second node.')
        # carriers
        pyomo_model.set_carriers = pe.Set(initialize=self.set_carriers, doc='Set of carriers')
        # technologies
        pyomo_model.set_technologies = pe.Set(initialize=self.set_technologies, doc='Set of technologies')
        # all elements
        pyomo_model.set_elements = pe.Set(initialize=pyomo_model.set_technologies | pyomo_model.set_carriers, doc='Set of elements')
        # set set_elements to indexing_sets
        self.indexing_sets.append("set_elements")
        # time-steps
        pyomo_model.set_base_time_steps = pe.Set(initialize=self.set_base_time_steps, doc='Set of base time-steps')
        # yearly time steps
        pyomo_model.set_time_steps_yearly = pe.Set(initialize=self.set_time_steps_yearly, doc='Set of yearly time-steps')
        # yearly time steps of entire optimization horizon
        pyomo_model.set_time_steps_yearly_entire_horizon = pe.Set(initialize=self.set_time_steps_yearly_entire_horizon, doc='Set of yearly time-steps of entire optimization horizon')

    def construct_params(self):
        """ constructs the pe.Params of the class <EnergySystem> """
        # carbon emissions limit
        cls = self.__class__
        parameters = self.optimization_setup.parameters
        pyomo_model = self.optimization_setup.model
        parameters.add_parameter(name="carbon_emissions_limit", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_limit", set_time_steps=pyomo_model.set_time_steps_yearly),
            doc='Parameter which specifies the total limit on carbon emissions')
        # carbon emissions budget
        parameters.add_parameter(name="carbon_emissions_budget", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_budget"),
            doc='Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon')
        # carbon emissions budget
        parameters.add_parameter(name="previous_carbon_emissions", data=self.optimization_setup.initialize_component(cls, "previous_carbon_emissions"), doc='Parameter which specifies the total previous carbon emissions')
        # carbon price
        parameters.add_parameter(name="carbon_price", data=self.optimization_setup.initialize_component(cls, "carbon_price", set_time_steps=pyomo_model.set_time_steps_yearly),
            doc='Parameter which specifies the yearly carbon price')
        # carbon price of overshoot
        parameters.add_parameter(name="carbon_price_overshoot", data=self.optimization_setup.initialize_component(cls, "carbon_price_overshoot"), doc='Parameter which specifies the carbon price for budget overshoot')

    def construct_vars(self):
        """ constructs the pe.Vars of the class <EnergySystem> """
        variables = self.optimization_setup.variables
        pyomo_model = self.optimization_setup.model
        # carbon emissions
        variables.add_variable(pyomo_model, name="carbon_emissions_total", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total carbon emissions of energy system")
        # cumulative carbon emissions
        variables.add_variable(pyomo_model, name="carbon_emissions_cumulative", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals,
            doc="cumulative carbon emissions of energy system over time for each year")
        # carbon emission overshoot
        variables.add_variable(pyomo_model, name="carbon_emissions_overshoot", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.NonNegativeReals,
            doc="overshoot carbon emissions of energy system at the end of the time horizon")
        # cost of carbon emissions
        variables.add_variable(pyomo_model, name="cost_carbon_emissions_total", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total cost of carbon emissions of energy system")
        # costs
        variables.add_variable(pyomo_model, name="cost_total", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total cost of energy system")
        # NPV
        variables.add_variable(pyomo_model, name="NPV", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="NPV of energy system")

    def construct_constraints(self):
        """ constructs the pe.Constraints of the class <EnergySystem> """
        constraints = self.optimization_setup.constraints
        pyomo_model = self.optimization_setup.model
        # carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_total", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_total_rule,
            doc="total carbon emissions of energy system")
        # carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_cumulative", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_cumulative_rule,
            doc="cumulative carbon emissions of energy system over time")
        # cost of carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_cost_total", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_cost_total_rule, doc="total carbon cost of energy system")
        # carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_limit", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_limit_rule,
            doc="limit of total carbon emissions of energy system")
        # carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_budget", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_budget_rule,
            doc="Budget of total carbon emissions of energy system")
        # costs
        constraints.add_constraint(pyomo_model, name="constraint_cost_total", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_cost_total_rule, doc="total cost of energy system")
        # NPV
        constraints.add_constraint(pyomo_model, name="constraint_NPV", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_NPV_rule, doc="NPV of energy system")


class EnergySystemRules:
    """
    This class takes care of the rules for the EnergySystem
    """

    def __init__(self, energy_system: EnergySystem):
        """
        Inits the constraints for a given energy syste,
        :param energy_system: The energy system used to build the constraints
        """

        self.energy_system = energy_system

    def constraint_carbon_emissions_total_rule(self, model, year):
        """ add up all carbon emissions from technologies and carriers """
        return (model.carbon_emissions_total[year] ==
                # technologies
                model.carbon_emissions_technology_total[year] + # carriers
                model.carbon_emissions_carrier_total[year])

    def constraint_carbon_emissions_cumulative_rule(self, model, year):
        """ cumulative carbon emissions over time """
        # get parameter object
        params = self.energy_system.parameters
        interval_between_years = self.energy_system.system["interval_between_years"]
        if year == model.set_time_steps_yearly.at(1):
            return (model.carbon_emissions_cumulative[year] == model.carbon_emissions_total[year] + params.previous_carbon_emissions)
        else:
            return (model.carbon_emissions_cumulative[year] == model.carbon_emissions_cumulative[year - 1] + model.carbon_emissions_total[year - 1] * (interval_between_years - 1) +
                    model.carbon_emissions_total[year])

    def constraint_carbon_cost_total_rule(self, model, year):
        """ carbon cost associated with the carbon emissions of the system in each year """
        # get parameter object
        params = self.energy_system.parameters
        return (model.cost_carbon_emissions_total[year] == params.carbon_price[year] * model.carbon_emissions_total[year] # add overshoot price
                + model.carbon_emissions_overshoot[year] * params.carbon_price_overshoot)

    def constraint_carbon_emissions_limit_rule(self, model, year):
        """ time dependent carbon emissions limit from technologies and carriers"""
        # get parameter object
        params = self.energy_system.parameters
        if params.carbon_emissions_limit[year] != np.inf:
            return (params.carbon_emissions_limit[year] >= model.carbon_emissions_total[year])
        else:
            return pe.Constraint.Skip

    def constraint_carbon_emissions_budget_rule(self, model, year):
        """ carbon emissions budget of entire time horizon from technologies and carriers.
        The prediction extends until the end of the horizon, i.e.,
        last optimization time step plus the current carbon emissions until the end of the horizon """
        # get parameter object
        params = self.energy_system.parameters
        interval_between_years = self.energy_system.system["interval_between_years"]
        if params.carbon_emissions_budget != np.inf:  # TODO check for last year - without last term?
            return (params.carbon_emissions_budget + model.carbon_emissions_overshoot[year] >= model.carbon_emissions_cumulative[year] + model.carbon_emissions_total[year] * (interval_between_years - 1))
        else:
            return pe.Constraint.Skip

    def constraint_cost_total_rule(self, model, year):
        """ add up all costs from technologies and carriers"""
        return (model.cost_total[year] ==
                # capex
                model.capex_total[year] + # opex
                model.opex_total[year] + # carrier costs
                model.cost_carrier_total[year] + # carbon costs
                model.cost_carbon_emissions_total[year])

    def constraint_NPV_rule(self, model, year):
        """ discounts the annual capital flows to calculate the NPV """
        system = self.energy_system.system
        discount_rate = self.energy_system.analysis["discount_rate"]
        if system["optimized_years"] > 1:
            interval_between_years = system["interval_between_years"]
        else:
            interval_between_years = 1

        return (model.NPV[year] == model.cost_total[year] * sum(# economic discount
            ((1 / (1 + discount_rate)) ** (interval_between_years * (year - model.set_time_steps_yearly.at(1)) + _intermediate_time_step)) for _intermediate_time_step in range(0, interval_between_years)))
