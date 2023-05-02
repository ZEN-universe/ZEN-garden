"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all transport technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for the transport technologies.
==========================================================================================================================================================================="""
import logging

import numpy as np
import pyomo.environ as pe

from .technology import Technology


class TransportTechnology(Technology):
    # set label
    label = "set_transport_technologies"
    location_type = "set_edges"

    def __init__(self, tech: str, optimization_setup):
        """init transport technology object
        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of """

        logging.info(f'Initialize transport technology {tech}')
        super().__init__(tech, optimization_setup)
        # dict of reversed edges
        self.dict_reversed_edges = {}
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # set attributes for parameters of child class <TransportTechnology>
        self.distance = self.data_input.extract_input_data("distance", index_sets=["set_edges"])
        self.transport_loss_factor = self.data_input.extract_attribute("transport_loss_factor")["value"]
        # get capex of transport technology
        self.get_capex_transport()
        # annualize capex
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()
        # check that existing capacities are equal in both directions if technology is bidirectional
        if self.name in self.optimization_setup.system["set_bidirectional_transport_technologies"]:
            self.check_if_bidirectional()

    def get_capex_transport(self):
        """get capex of transport technology"""
        set_time_steps_yearly = self.energy_system.set_time_steps_yearly
        # check if there are separate capex for capacity and distance
        if self.optimization_setup.system['double_capex_transport']:
            # both capex terms must be specified
            self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
            self.capex_per_distance_transport = self.data_input.extract_input_data("capex_per_distance_transport", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        else:  # Here only capex_specific is used, and capex_per_distance_transport is set to Zero.
            if self.data_input.exists_attribute("capex_per_distance_transport"):
                self.capex_per_distance_transport = self.data_input.extract_input_data("capex_per_distance_transport", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
                self.capex_specific = self.capex_per_distance_transport * self.distance
                self.opex_specific_fixed = self.opex_specific_fixed * self.distance
            elif self.data_input.exists_attribute("capex_specific"):
                self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
            else:
                raise AttributeError(f"The transport technology {self.name} has neither capex_per_distance_transport nor capex_specific attribute.")
            self.capex_per_distance_transport = self.capex_specific * 0.0

    def convert_to_fraction_of_capex(self):
        """ this method converts the total capex to fraction of capex, depending on how many hours per year are calculated """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.capex_specific = self.capex_specific * fraction_year
        self.capex_per_distance_transport = self.capex_per_distance_transport * fraction_year

    def calculate_capex_of_single_capacity(self, capacity, index):
        """ this method calculates the annualized capex of a single existing capacity. """
        # TODO check existing capex of transport techs -> Hannes
        if np.isnan(self.capex_specific[index[0]].iloc[0]):
            return 0
        else:
            return self.capex_specific[index[0]].iloc[0] * capacity

    def check_if_bidirectional(self):
        """ checks that the existing capacities in both directions of bidirectional capacities are equal """
        for edge in self.energy_system.set_edges:
            _reversed_edge = self.energy_system.calculate_reversed_edge(edge)
            self.set_reversed_edge(edge=edge, _reversed_edge=_reversed_edge)
            _capacity_existing_edge = self.capacity_existing[edge]
            _capacity_existing_reversed_edge = self.capacity_existing[_reversed_edge]
            assert (
                        _capacity_existing_edge == _capacity_existing_reversed_edge).all(), f"The existing capacities of the bidirectional transport technology {self.name} are not equal on the edge pair {edge} and {_reversed_edge} ({_capacity_existing_edge.to_dict()} and {_capacity_existing_reversed_edge.to_dict()})"

    ### --- getter/setter classmethods
    def set_reversed_edge(self, edge, _reversed_edge):
        """ maps the reversed edge to an edge """
        self.dict_reversed_edges[edge] = _reversed_edge

    def get_reversed_edge(self, edge):
        """ get the reversed edge corresponding to an edge """
        return self.dict_reversed_edges[edge]

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to TransportTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <TransportTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        pass

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <TransportTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """

        # distance between nodes
        optimization_setup.parameters.add_parameter(name="distance", data=optimization_setup.initialize_component(cls, "distance", index_names=["set_transport_technologies", "set_edges"]),
            doc='distance between two nodes for transport technologies')
        # capital cost per unit
        optimization_setup.parameters.add_parameter(name="capex_specific_transport",
            data=optimization_setup.initialize_component(cls, "capex_specific", index_names=["set_transport_technologies", "set_edges", "set_time_steps_yearly"]),
            doc='capex per unit for transport technologies')
        # capital cost per distance
        optimization_setup.parameters.add_parameter(name="capex_per_distance_transport",
            data=optimization_setup.initialize_component(cls, 'capex_per_distance_transport', index_names=['set_transport_technologies', "set_edges", "set_time_steps_yearly"]),
            doc='capex per distance for transport technologies')
        # carrier losses
        optimization_setup.parameters.add_parameter(name="transport_loss_factor", data=optimization_setup.initialize_component(cls, "transport_loss_factor", index_names=["set_transport_technologies"]),
            doc='carrier losses due to transport with transport technologies')

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <TransportTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """

        def flow_transport_bounds(model, tech, edge, time):
            """ return bounds of flow_transport for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param edge: edge index
            :param time: time index
            :return bounds: bounds of flow_transport"""
            # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
            time_step_year = optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech, time)
            bounds = model.capacity[tech, "power", edge, time_step_year].bounds
            return (bounds)

        model = optimization_setup.model
        # flow of carrier on edge
        optimization_setup.variables.add_variable(model, name="flow_transport", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"], optimization_setup), domain=pe.NonNegativeReals,
            bounds=flow_transport_bounds, doc='carrier flow through transport technology on edge i and time t')
        # loss of carrier on edge
        optimization_setup.variables.add_variable(model, name="flow_transport_loss", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"], optimization_setup), domain=pe.NonNegativeReals,
            doc='carrier flow through transport technology on edge i and time t')

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <TransportTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        rules = TransportTechnologyRules(optimization_setup)
        # Carrier Flow Losses 
        optimization_setup.constraints.add_constraint(model, name="constraint_transport_technology_losses_flow", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"], optimization_setup),
            rule=rules.constraint_transport_technology_losses_flow_rule, doc='Carrier loss due to transport with through transport technology')
        # capex of transport technologies
        optimization_setup.constraints.add_constraint(model, name="constraint_transport_technology_capex", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_transport_technology_capex_rule, doc='Capital expenditures for installing transport technology')
        # bidirectional transport technologies: capacity on edge must be equal in both directions
        optimization_setup.constraints.add_constraint(model, name="constraint_transport_technology_bidirectional", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_transport_technology_bidirectional_rule, doc='Forces that transport technology capacity must be equal in both directions')

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls, optimization_setup, disjunct, tech, capacity_type, edge, time):
        """definition of disjunct constraints if technology is on"""
        model = disjunct.model()
        # get parameter object
        params = optimization_setup.parameters
        # get invest time step
        time_step_year = optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech, time)
        # disjunct constraints min load
        disjunct.constraint_min_load = pe.Constraint(
            expr=model.flow_transport[tech, edge, time] >= params.min_load[tech, capacity_type, edge, time] * model.capacity[tech, capacity_type, edge, time_step_year])

    @classmethod
    def disjunct_off_technology_rule(cls,optimization_setup, disjunct, tech, capacity_type, edge, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraint_no_load = pe.Constraint(expr=model.flow_transport[tech, edge, time] == 0)


class TransportTechnologyRules:
    """
    Rules for the TransportTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """
        self.optimization_setup = optimization_setup

    ### --- functions with constraint rules --- ###
    def constraint_transport_technology_losses_flow_rule(self, model, tech, edge, time):
        """compute the flow losses for a carrier through a transport technology"""
        # get parameter object
        params = self.optimization_setup.parameters
        if np.isinf(params.distance[tech, edge]):
            return model.flow_transport_loss[tech, edge, time] == 0
        else:
            return (model.flow_transport_loss[tech, edge, time] == params.distance[tech, edge] * params.transport_loss_factor[tech] * model.flow_transport[tech, edge, time])

    def constraint_transport_technology_capex_rule(self, model, tech, edge, time):
        """ definition of the capital expenditures for the transport technology"""
        # get parameter object
        params = self.optimization_setup.parameters
        if np.isinf(params.distance[tech, edge]):
            return model.capacity_addition[tech, "power", edge, time] == 0
        else:
            return (model.cost_capex[tech, "power", edge, time] == model.capacity_addition[tech, "power", edge, time] * params.capex_specific_transport[tech, edge, time] + model.technology_installation[
                tech, "power", edge, time] * params.distance[tech, edge] * params.capex_per_distance_transport[tech, edge, time])

    def constraint_transport_technology_bidirectional_rule(self, model, tech, edge, time):
        """ Forces that transport technology capacity must be equal in both direction"""
        system = self.optimization_setup.system
        if tech in system["set_bidirectional_transport_technologies"]:
            _reversed_edge = self.get_reversed_edge(edge)
            return (model.capacity_addition[tech, "power", edge, time] == model.capacity_addition[tech, "power", _reversed_edge, time])
        else:
            return pe.Constraint.Skip
