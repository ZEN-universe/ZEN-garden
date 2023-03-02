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
        self.loss_flow = self.data_input.extract_attribute("loss_flow")["value"]
        # get capex of transport technology
        self.get_capex_transport()
        # annualize capex
        self.convert_to_annualized_capex()
        # calculate capex of existing capacity
        self.capex_existing_capacity = self.calculate_capex_of_existing_capacities()
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
            self.capex_per_distance = self.data_input.extract_input_data("capex_per_distance", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        else:  # Here only capex_specific is used, and capex_per_distance is set to Zero.
            if self.data_input.exists_attribute("capex_per_distance"):
                self.capex_per_distance = self.data_input.extract_input_data("capex_per_distance", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
                self.capex_specific = self.capex_per_distance * self.distance
                self.fixed_opex_specific = self.fixed_opex_specific * self.distance
            elif self.data_input.exists_attribute("capex_specific"):
                self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
            else:
                raise AttributeError(f"The transport technology {self.name} has neither capex_per_distance nor capex_specific attribute.")
            self.capex_per_distance = self.capex_specific * 0.0

    def convert_to_annualized_capex(self):
        """ this method converts the total capex to annualized capex """
        fractional_annuity = self.calculate_fractional_annuity()
        _fraction_year = self.optimization_setup.system["unaggregated_time_steps_per_year"] / self.optimization_setup.system["total_hours_per_year"]
        # annualize capex
        self.capex_specific = self.capex_specific * fractional_annuity + self.fixed_opex_specific * _fraction_year
        self.capex_per_distance = self.capex_per_distance * fractional_annuity

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
            _existing_capacity_edge = self.existing_capacity[edge]
            _existing_capacity_reversed_edge = self.existing_capacity[_reversed_edge]
            assert (
                        _existing_capacity_edge == _existing_capacity_reversed_edge).all(), f"The existing capacities of the bidirectional transport technology {self.name} are not equal on the edge pair {edge} and {_reversed_edge} ({_existing_capacity_edge.to_dict()} and {_existing_capacity_reversed_edge.to_dict()})"

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
        optimization_setup.parameters.add_parameter(name="capex_per_distance",
            data=optimization_setup.initialize_component(cls, 'capex_per_distance', index_names=['set_transport_technologies', "set_edges", "set_time_steps_yearly"]),
            doc='capex per distance for transport technologies')
        # carrier losses
        optimization_setup.parameters.add_parameter(name="loss_flow", data=optimization_setup.initialize_component(cls, "loss_flow", index_names=["set_transport_technologies"]),
            doc='carrier losses due to transport with transport technologies')

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <TransportTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """

        def carrier_flow_bounds(model, tech, edge, time):
            """ return bounds of carrier_flow for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param edge: edge index
            :param time: time index
            :return bounds: bounds of carrier_flow"""
            # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
            time_step_year = optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech, time)
            bounds = model.capacity[tech, "power", edge, time_step_year].bounds
            return (bounds)

        model = optimization_setup.model
        # flow of carrier on edge
        optimization_setup.variables.add_variable(model, name="carrier_flow", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"], optimization_setup), domain=pe.NonNegativeReals,
            bounds=carrier_flow_bounds, doc='carrier flow through transport technology on edge i and time t')
        # loss of carrier on edge
        optimization_setup.variables.add_variable(model, name="carrier_loss", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"], optimization_setup), domain=pe.NonNegativeReals,
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
            expr=model.carrier_flow[tech, edge, time] >= params.min_load[tech, capacity_type, edge, time] * model.capacity[tech, capacity_type, edge, time_step_year])

    @classmethod
    def disjunct_off_technology_rule(cls,optimization_setup, disjunct, tech, capacity_type, edge, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraint_no_load = pe.Constraint(expr=model.carrier_flow[tech, edge, time] == 0)


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
            return model.carrier_loss[tech, edge, time] == 0
        else:
            return (model.carrier_loss[tech, edge, time] == params.distance[tech, edge] * params.loss_flow[tech] * model.carrier_flow[tech, edge, time])

    def constraint_transport_technology_capex_rule(self, model, tech, edge, time):
        """ definition of the capital expenditures for the transport technology"""
        # get parameter object
        params = self.optimization_setup.parameters
        if np.isinf(params.distance[tech, edge]):
            return model.built_capacity[tech, "power", edge, time] == 0
        else:
            return (model.capex[tech, "power", edge, time] == model.built_capacity[tech, "power", edge, time] * params.capex_specific_transport[tech, edge, time] + model.install_technology[
                tech, "power", edge, time] * params.distance[tech, edge] * params.capex_per_distance[tech, edge, time])

    def constraint_transport_technology_bidirectional_rule(self, model, tech, edge, time):
        """ Forces that transport technology capacity must be equal in both direction"""
        system = self.optimization_setup.system
        if tech in system["set_bidirectional_transport_technologies"]:
            _reversed_edge = self.get_reversed_edge(edge)
            return (model.built_capacity[tech, "power", edge, time] == model.built_capacity[tech, "power", _reversed_edge, time])
        else:
            return pe.Constraint.Skip
