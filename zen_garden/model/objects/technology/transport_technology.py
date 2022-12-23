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
import warnings
import pyomo.environ as pe
import pyomo.gdp as pgdp
import numpy as np
from .technology import Technology
from .conversion_technology import ConversionTechnology
from ..energy_system import EnergySystem
from ..component import Parameter, Variable, Constraint


class TransportTechnology(Technology):
    # set label
    label = "set_transport_technologies"
    location_type = "set_edges"
    # empty list of elements
    list_of_elements = []
    # dict of reversed edges
    dict_reversed_edges = {}

    def __init__(self, tech):
        """init transport technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize transport technology {tech}')
        super().__init__(tech)
        # store input data
        self.store_input_data()
        # add TransportTechnology to list
        TransportTechnology.add_element(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # set attributes for parameters of child class <TransportTechnology>
        self.distance = self.datainput.extract_input_data("distance", index_sets=["set_edges"])
        self.loss_flow = self.datainput.extract_attribute("loss_flow")["value"]
        # get capex of transport technology
        self.get_capex_transport()
        # annualize capex
        self.convert_to_annualized_capex()
        # calculate capex of existing capacity
        self.capex_existing_capacity = self.calculate_capex_of_existing_capacities()
        # check that existing capacities are equal in both directions if technology is bidirectional
        if self.name in EnergySystem.get_system()["set_bidirectional_transport_technologies"]:
            self.check_if_bidirectional()

    def get_capex_transport(self):
        """get capex of transport technology"""
        set_time_steps_yearly = EnergySystem.get_energy_system().set_time_steps_yearly
        # check if there are separate capex for capacity and distance
        if EnergySystem.system['double_capex_transport']:
            # both capex terms must be specified
            self.capex_specific = self.datainput.extract_input_data("capex_specific", index_sets=["set_edges", "set_time_steps"], time_steps=set_time_steps_yearly)
            self.capex_per_distance = self.datainput.extract_input_data("capex_per_distance", index_sets=["set_edges", "set_time_steps"], time_steps=set_time_steps_yearly)
        else:  # Here only capex_specific is used, and capex_per_distance is set to Zero.
            if self.datainput.exists_attribute("capex_per_distance"):
                self.capex_per_distance = self.datainput.extract_input_data("capex_per_distance", index_sets=["set_edges", "set_time_steps"], time_steps=set_time_steps_yearly)
                self.capex_specific = self.capex_per_distance * self.distance
                self.fixed_opex_specific = self.fixed_opex_specific * self.distance
            elif self.datainput.exists_attribute("capex_specific"):
                self.capex_specific = self.datainput.extract_input_data("capex_specific", index_sets=["set_edges", "set_time_steps"], time_steps=set_time_steps_yearly)
            else:
                raise AttributeError(f"The transport technology {self.name} has neither capex_per_distance nor capex_specific attribute.")
            self.capex_per_distance = self.capex_specific * 0.0

    def convert_to_annualized_capex(self):
        """ this method converts the total capex to annualized capex """
        fractional_annuity = self.calculate_fractional_annuity()
        system = EnergySystem.get_system()
        _fraction_year = system["unaggregated_time_steps_per_year"] / system["total_hours_per_year"]
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
        energy_system = EnergySystem.get_energy_system()
        for edge in energy_system.set_edges:
            _reversed_edge = EnergySystem.calculate_reversed_edge(edge)
            TransportTechnology.set_reversed_edge(edge=edge, _reversed_edge=_reversed_edge)
            _existing_capacity_edge = self.existing_capacity[edge]
            _existing_capacity_reversed_edge = self.existing_capacity[_reversed_edge]
            assert (
                        _existing_capacity_edge == _existing_capacity_reversed_edge).all(), f"The existing capacities of the bidirectional transport technology {self.name} are not equal on the edge pair {edge} and {_reversed_edge} ({_existing_capacity_edge.to_dict()} and {_existing_capacity_reversed_edge.to_dict()})"

    ### --- getter/setter classmethods
    @classmethod
    def set_reversed_edge(cls, edge, _reversed_edge):
        """ maps the reversed edge to an edge """
        cls.dict_reversed_edges[edge] = _reversed_edge

    @classmethod
    def getReversedEdge(cls, edge):
        """ get the reversed edge corresponding to an edge """
        return cls.dict_reversed_edges[edge]

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to TransportTechnology --- ###
    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <TransportTechnology> """
        pass

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <TransportTechnology> """
        model = EnergySystem.get_pyomo_model()

        # distance between nodes
        Parameter.add_parameter(name="distance", data=EnergySystem.initialize_component(cls, "distance", index_names=["set_transport_technologies", "set_edges"]),
            doc='distance between two nodes for transport technologies')
        # capital cost per unit
        Parameter.add_parameter(name="capex_specific_transport",
            data=EnergySystem.initialize_component(cls, "capex_specific", index_names=["set_transport_technologies", "set_edges", "set_time_steps_yearly"]),
            doc='capex per unit for transport technologies')
        # capital cost per distance
        Parameter.add_parameter(name="capex_per_distance",
            data=EnergySystem.initialize_component(cls, 'capex_per_distance', index_names=['set_transport_technologies', "set_edges", "set_time_steps_yearly"]),
            doc='capex per distance for transport technologies')
        # carrier losses
        Parameter.add_parameter(name="loss_flow", data=EnergySystem.initialize_component(cls, "loss_flow", index_names=["set_transport_technologies"]),
            doc='carrier losses due to transport with transport technologies')

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <TransportTechnology> """

        def carrier_flow_bounds(model, tech, edge, time):
            """ return bounds of carrier_flow for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param edge: edge index
            :param time: time index
            :return bounds: bounds of carrier_flow"""
            # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
            time_step_year = EnergySystem.convert_time_step_operation2invest(tech, time)
            bounds = model.capacity[tech, "power", edge, time_step_year].bounds
            return (bounds)

        model = EnergySystem.get_pyomo_model()
        # flow of carrier on edge
        Variable.add_variable(model, name="carrier_flow", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"]), domain=pe.NonNegativeReals,
            bounds=carrier_flow_bounds, doc='carrier flow through transport technology on edge i and time t')
        # loss of carrier on edge
        Variable.add_variable(model, name="carrier_loss", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"]), domain=pe.NonNegativeReals,
            doc='carrier flow through transport technology on edge i and time t')

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <TransportTechnology> """
        model = EnergySystem.get_pyomo_model()
        system = EnergySystem.get_system()
        # Carrier Flow Losses 
        Constraint.add_constraint(model, name="constraint_transport_technology_losses_flow", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"]),
            rule=constraint_transport_technology_losses_flow_rule, doc='Carrier loss due to transport with through transport technology')
        # capex of transport technologies
        Constraint.add_constraint(model, name="constraint_transport_technology_capex", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_yearly"]),
            rule=constraint_transport_technology_capex_rule, doc='Capital expenditures for installing transport technology')
        # bidirectional transport technologies: capacity on edge must be equal in both directions
        Constraint.add_constraint(model, name="constraint_transport_technology_bidirectional", index_sets=cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_yearly"]),
            rule=constraint_transport_technology_bidirectional_rule, doc='Forces that transport technology capacity must be equal in both directions')
        # disjunct to enforce Selfish behavior
        if "enforce_selfish_behavior" in system.keys() and system["enforce_selfish_behavior"]:
            Constraint.add_constraint(model,"disjunct_selfish_behavior_no_flow",index_sets=cls.create_custom_set(["set_transport_technologies", "set_selfish_nodes"]),
                                      rule=cls.disjunct_selfish_behavior_no_flow_rule,doc="disjunct to enforce Selfish behavior no flow",constraint_class=pgdp.Disjunct)
            Constraint.add_constraint(model,"disjunct_selfish_behavior_no_shed_demand_low",index_sets=cls.create_custom_set(["set_transport_technologies", "set_selfish_nodes"]),
                                      rule=cls.disjunct_selfish_behavior_no_shed_demand_low_rule,doc="disjunct to enforce Selfish behavior no shed demand at low cost",constraint_class=pgdp.Disjunct)
            Constraint.add_constraint(model,"disjunction_selfish_behavior",index_sets=cls.create_custom_set(["set_transport_technologies", "set_selfish_nodes"]),
                                      rule=cls.disjunction_selfish_behavior_rule,doc="disjunction to enforce Selfish behavior",constraint_class=pgdp.Disjunction)

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls, disjunct, tech, capacity_type, edge, time):
        """definition of disjunct constraints if technology is on"""
        model = disjunct.model()
        # get parameter object
        params = Parameter.get_component_object()
        # get invest time step
        time_step_year = EnergySystem.convert_time_step_operation2invest(tech, time)
        # disjunct constraints min load
        disjunct.constraint_min_load = pe.Constraint(
            expr=model.carrier_flow[tech, edge, time] >= params.min_load[tech, capacity_type, edge, time] * model.capacity[tech, capacity_type, edge, time_step_year])

    @classmethod
    def disjunct_off_technology_rule(cls, disjunct, tech, capacity_type, edge, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraint_no_load = pe.Constraint(expr=model.carrier_flow[tech, edge, time] == 0)


    @classmethod
    def disjunct_selfish_behavior_no_flow_rule(cls,disjunct,tech,node):
        """ definition of disjunct constraint to enforce that reducing own voluntarily shed demand is preferred over transporting to other nodes - no flow"""
        model       = disjunct.model()
        solver      = EnergySystem.get_solver()
        edges_out    = EnergySystem.calculate_connected_edges(node,direction="out")
        disjunct.constraint_no_flow_out = pe.Constraint(
            expr= sum(
                sum(
                    # sum(
                    model.carrier_flow[tech, edge, time]
                    for time in model.set_time_steps_operation[tech]
                    # )
                    # for tech in model.set_transport_technologies
                )
                for edge in edges_out
            ) <= 10**(-solver["rounding_decimal_points_ts"])
            # ) == 0
        )

    @classmethod
    def disjunct_selfish_behavior_no_shed_demand_low_rule(cls, disjunct,tech, node):
        """ definition of disjunct constraint to enforce that reducing own voluntarily shed demand is preferred over transporting to other nodes - no flow"""
        model   = disjunct.model()
        solver  = EnergySystem.get_solver()
        # set shedDemandCarrierLow of all carriers that are either transported (reference_carrier)
        # or that are the output of conversionTechnologies with the reference_carrier of this transportTechnology as the inputCarrier
        list_connected_carriers = []
        # for tech in model.set_transport_technologies:
        reference_carrier            = cls.get_attribute_of_specific_element(tech,"reference_carrier")[0]
        list_connected_carriers.extend([reference_carrier])
        set_conversion_technologies   = EnergySystem.get_energy_system().set_conversion_technologies
        for conversion_technology in set_conversion_technologies:
            if reference_carrier in ConversionTechnology.get_attribute_of_specific_element(conversion_technology,"input_carrier"):
                list_connected_carriers.extend(ConversionTechnology.get_attribute_of_specific_element(conversion_technology,"output_carrier"))
        list_unique_connected_carriers = list(set(list_connected_carriers))
        disjunct.constraint_no_shed_demand_carrier_low = pe.Constraint(
            expr= sum(
                sum(
                    model.shed_demand_carrier[connected_carrier, node, time]
                    for time in model.set_time_steps_operation[connected_carrier]
                )
                for connected_carrier in list_unique_connected_carriers
            ) <= 10**(-solver["rounding_decimal_points_ts"])
            # ) == 0
        )

    @classmethod
    def disjunction_selfish_behavior_rule(cls,model,tech,node):
        """ definition that enforces Selfish behavior disjuncts """
        return([model.disjunct_selfish_behavior_no_flow[tech,node],model.disjunct_selfish_behavior_no_shed_demand_low[tech,node]])

### --- functions with constraint rules --- ###
def constraint_transport_technology_losses_flow_rule(model, tech, edge, time):
    """compute the flow losses for a carrier through a transport technology"""
    # get parameter object
    params = Parameter.get_component_object()
    if np.isinf(params.distance[tech, edge]):
        return model.carrier_loss[tech, edge, time] == 0
    else:
        return (model.carrier_loss[tech, edge, time] == params.distance[tech, edge] * params.loss_flow[tech] * model.carrier_flow[tech, edge, time])


def constraint_transport_technology_capex_rule(model, tech, edge, time):
    """ definition of the capital expenditures for the transport technology"""
    # get parameter object
    params = Parameter.get_component_object()
    if np.isinf(params.distance[tech, edge]):
        return model.built_capacity[tech, "power", edge, time] == 0
    else:
        return (model.capex[tech, "power", edge, time] == model.built_capacity[tech, "power", edge, time] * params.capex_specific_transport[tech, edge, time] + model.install_technology[
            tech, "power", edge, time] * params.distance[tech, edge] * params.capex_per_distance[tech, edge, time])


def constraint_transport_technology_bidirectional_rule(model, tech, edge, time):
    """ Forces that transport technology capacity must be equal in both direction"""
    system = EnergySystem.get_system()
    if tech in system["set_bidirectional_transport_technologies"]:
        _reversed_edge = TransportTechnology.getReversedEdge(edge)
        return (model.built_capacity[tech, "power", edge, time] == model.built_capacity[tech, "power", _reversed_edge, time])
    else:
        return pe.Constraint.Skip
