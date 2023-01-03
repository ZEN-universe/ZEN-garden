"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining a generic energy carrier.
                The class takes as inputs the abstract optimization model. The class adds parameters, variables and
                constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import numpy as np
import pandas as pd
from ..element import Element
from ..energy_system import EnergySystem
from ..technology.technology import Technology
from ..component import Parameter, Variable, Constraint


class Carrier(Element):
    # set label
    label = "set_carriers"
    # empty list of elements
    list_of_elements = []

    def __init__(self, carrier):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model"""

        logging.info(f'Initialize carrier {carrier}')
        super().__init__(carrier)
        # store input data
        self.store_input_data()
        # add carrier to list
        Carrier.add_element(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        set_base_time_steps_yearly = EnergySystem.get_energy_system().set_base_time_steps_yearly
        set_time_steps_yearly = EnergySystem.get_energy_system().set_time_steps_yearly
        # set attributes of carrier
        # raw import
        self.raw_time_series = {}
        self.raw_time_series["demand_carrier"] = self.datainput.extract_input_data("demand_carrier", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availability_carrier_import"] = self.datainput.extract_input_data("availability_carrier", index_sets=["set_nodes", "set_time_steps"],
                                                                                                column="availability_carrier_import", time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availability_carrier_export"] = self.datainput.extract_input_data("availability_carrier", index_sets=["set_nodes", "set_time_steps"],
                                                                                                column="availability_carrier_export", time_steps=set_base_time_steps_yearly)
        self.raw_time_series["export_price_carrier"] = self.datainput.extract_input_data("price_carrier", index_sets=["set_nodes", "set_time_steps"], column="export_price_carrier",
                                                                                         time_steps=set_base_time_steps_yearly)
        self.raw_time_series["import_price_carrier"] = self.datainput.extract_input_data("price_carrier", index_sets=["set_nodes", "set_time_steps"], column="import_price_carrier",
                                                                                         time_steps=set_base_time_steps_yearly)
        # non-time series input data
        self.availability_carrier_import_yearly = self.datainput.extract_input_data("availability_carrier_yearly", index_sets=["set_nodes", "set_time_steps"],
                                                                                    column="availability_carrier_import_yearly", time_steps=set_time_steps_yearly)
        self.availability_carrier_export_yearly = self.datainput.extract_input_data("availability_carrier_yearly", index_sets=["set_nodes", "set_time_steps"],
                                                                                    column="availability_carrier_export_yearly", time_steps=set_time_steps_yearly)
        self.carbon_intensity_carrier = self.datainput.extract_input_data("carbon_intensity", index_sets=["set_nodes", "set_time_steps"], time_steps=set_time_steps_yearly)
        self.shed_demand_price = self.datainput.extract_input_data("shed_demand_price", index_sets=[])
        self.shed_demand_price_high = self.datainput.extract_input_data("shed_demand_price_high", index_sets=[])
        self.max_shed_demand = self.datainput.extract_input_data("max_shed_demand", index_sets=[])
        self.max_shed_demand_high = self.datainput.extract_input_data("max_shed_demand_high",index_sets = [])

    def overwrite_time_steps(self, base_time_steps):
            """ overwrites set_time_steps_operation"""
            set_time_steps_operation = EnergySystem.encode_time_step(self.name, base_time_steps=base_time_steps, time_step_type="operation", yearly=True)
            setattr(self, "set_time_steps_operation", set_time_steps_operation.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <Carrier> """
        pass

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <Carrier> """
        # demand of carrier
        Parameter.add_parameter(name="demand_carrier", data=EnergySystem.initialize_component(cls, "demand_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the carrier demand')
        # availability of carrier
        Parameter.add_parameter(name="availability_carrier_import",
            data=EnergySystem.initialize_component(cls, "availability_carrier_import", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries')
        # availability of carrier
        Parameter.add_parameter(name="availability_carrier_export",
            data=EnergySystem.initialize_component(cls, "availability_carrier_export", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries')
        # availability of carrier
        Parameter.add_parameter(name="availability_carrier_import_yearly",
            data=EnergySystem.initialize_component(cls, "availability_carrier_import_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year')
        # availability of carrier
        Parameter.add_parameter(name="availability_carrier_export_yearly",
            data=EnergySystem.initialize_component(cls, "availability_carrier_export_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year')
        # import price
        Parameter.add_parameter(name="import_price_carrier", data=EnergySystem.initialize_component(cls, "import_price_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the import carrier price')
        # export price
        Parameter.add_parameter(name="export_price_carrier", data=EnergySystem.initialize_component(cls, "export_price_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the export carrier price')
        # demand shedding price low
        Parameter.add_parameter(name="shed_demand_price", data=EnergySystem.initialize_component(cls, "shed_demand_price", index_names=["set_carriers"]),
            doc='Parameter which specifies the price to shed demand')
        # demand shedding price high
        Parameter.add_parameter(name="shed_demand_price_high", data=EnergySystem.initialize_component(cls, "shed_demand_price_high", index_names=["set_carriers"]),
            doc='Parameter which specifies the high price to shed demand')
        # max shed demand low
        Parameter.add_parameter(name="max_shed_demand", data=EnergySystem.initialize_component(cls, "max_shed_demand", index_names=["set_carriers"]),
            doc='Parameter which specifies the maximum fraction of shed demand at low price')
        # max shed demand high
        Parameter.add_parameter(name="max_shed_demand_high", data=EnergySystem.initialize_component(cls, "max_shed_demand_high", index_names=["set_carriers"]),
            doc='Parameter which specifies the maximum fraction of shed demand at high price')
        # carbon intensity
        Parameter.add_parameter(name="carbon_intensity_carrier",
            data=EnergySystem.initialize_component(cls, "carbon_intensity_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the carbon intensity of carrier')

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <Carrier> """
        def shed_demand_carrier_bounds(model,carrier,node, time):
            """ return bounds of shed demand carrier for bigM expression
            :param model: pe.ConcreteModel
            :param carrier: carrier index
            :param node: node
            :param time: operational time step
            :return bounds: bounds of shedDemandCarrierLow"""
            # bounds only needed for Big-M formulation, if enforce_selfish_behavior
            system = EnergySystem.get_system()
            if "enforce_selfish_behavior" in system.keys() and system["enforce_selfish_behavior"]:
                params = Parameter.get_component_object()
                demand_carrier = params.demand_carrier[carrier,node,time]
                bounds = (0,demand_carrier)
                return(bounds)
            else:
                return(None,None)

        model = EnergySystem.get_pyomo_model()

        # flow of imported carrier
        Variable.add_variable(model, name="import_carrier_flow", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.NonNegativeReals,
            doc='node- and time-dependent carrier import from the grid')
        # flow of exported carrier
        Variable.add_variable(model, name="export_carrier_flow", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.NonNegativeReals,
            doc='node- and time-dependent carrier export from the grid')
        # carrier import/export cost
        Variable.add_variable(model, name="cost_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.Reals,
            doc='node- and time-dependent carrier cost due to import and export')
        # total carrier import/export cost
        Variable.add_variable(model, name="cost_carrier_total", index_sets=model.set_time_steps_yearly, domain=pe.Reals, doc='total carrier cost due to import and export')
        # carbon emissions
        Variable.add_variable(model, name="carbon_emissions_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.Reals,
            doc="carbon emissions of importing and exporting carrier")
        # carbon emissions carrier
        Variable.add_variable(model, name="carbon_emissions_carrier_total", index_sets=model.set_time_steps_yearly, domain=pe.Reals, doc="total carbon emissions of importing and exporting carrier")
        # shed demand
        Variable.add_variable(model, name="shed_demand_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.NonNegativeReals,bounds=shed_demand_carrier_bounds,
            doc="shed demand of carrier")
        # shed demand high
        Variable.add_variable(model, name="shed_demand_carrier_high", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.NonNegativeReals,bounds=shed_demand_carrier_bounds,
            doc="shed demand of carrier at high price")
        # cost of shed demand
        Variable.add_variable(model, name="cost_shed_demand_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.NonNegativeReals,
            doc="cost of shedding demand of carrier")
        # cost of shed demand high
        Variable.add_variable(model, name="cost_shed_demand_carrier_high", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), domain=pe.NonNegativeReals,
            doc="cost of shedding demand of carrier at high price")

        # add pe.Sets of the child classes
        for subclass in cls.get_all_subclasses():
            if np.size(EnergySystem.get_system()[subclass.label]):
                subclass.construct_vars()

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.get_pyomo_model()

        # limit import flow by availability
        Constraint.add_constraint(model, name="constraint_availability_carrier_import", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_availability_carrier_import_rule, doc='node- and time-dependent carrier availability to import from outside the system boundaries', )
        # limit export flow by availability
        Constraint.add_constraint(model, name="constraint_availability_carrier_export", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_availability_carrier_export_rule, doc='node- and time-dependent carrier availability to export to outside the system boundaries', )
        # limit import flow by availability for each year
        Constraint.add_constraint(model, name="constraint_availability_carrier_import_yearly", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            rule=constraint_availability_carrier_import_yearly_rule, doc='node- and time-dependent carrier availability to import from outside the system boundaries summed over entire year', )
        # limit export flow by availability for each year
        Constraint.add_constraint(model, name="constraint_availability_carrier_export_yearly", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            rule=constraint_availability_carrier_export_yearly_rule, doc='node- and time-dependent carrier availability to export to outside the system boundaries summed over entire year', )
        # cost for carrier
        Constraint.add_constraint(model, name="constraint_cost_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]), rule=constraint_cost_carrier_rule,
            doc="cost of importing and exporting carrier")
        # cost for shed demand
        Constraint.add_constraint(model, name="constraint_cost_shed_demand", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_cost_shed_demand_rule, doc="cost of shedding carrier demand")
        # cost for shed demand high
        Constraint.add_constraint(model, name="constraint_cost_shed_demand_high", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_cost_shed_demand_high_rule, doc="cost of shedding carrier demand at high price")
        # limit of shed demand
        Constraint.add_constraint(model, name="constraint_limit_shed_demand", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_limit_shed_demand_rule, doc="limit of shedding carrier demand")
        # limit of shed demand
        Constraint.add_constraint(model, name="constraint_limit_shed_demand_high", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_limit_shed_demand_high_rule, doc="limit of shedding carrier demand a high price")
        # total cost for carriers
        Constraint.add_constraint(model, name="constraint_cost_carrier_total", index_sets=model.set_time_steps_yearly, rule=constraint_cost_carrier_total_rule,
            doc="total cost of importing and exporting carriers")
        # carbon emissions
        Constraint.add_constraint(model, name="constraint_carbon_emissions_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_carbon_emissions_carrier_rule, doc="carbon emissions of importing and exporting carrier")

        # carbon emissions carrier
        Constraint.add_constraint(model, name="constraint_carbon_emissions_carrier_total", index_sets=model.set_time_steps_yearly, rule=constraint_carbon_emissions_carrier_total_rule,
            doc="total carbon emissions of importing and exporting carriers")
        # energy balance
        Constraint.add_constraint(model, name="constraint_nodal_energy_balance", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
            rule=constraint_nodal_energy_balance_rule, doc='node- and time-dependent energy balance for each carrier', )
        # add pe.Sets of the child classes
        for subclass in cls.get_all_subclasses():
            if np.size(EnergySystem.get_system()[subclass.label]):
                subclass.construct_constraints()


# %% Constraint rules defined in current class
def constraint_availability_carrier_import_rule(model, carrier, node, time):
    """node- and time-dependent carrier availability to import from outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    if params.availability_carrier_import[carrier, node, time] != np.inf:
        return (model.import_carrier_flow[carrier, node, time] <= params.availability_carrier_import[carrier, node, time])
    else:
        return pe.Constraint.Skip


def constraint_availability_carrier_export_rule(model, carrier, node, time):
    """node- and time-dependent carrier availability to export to outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    if params.availability_carrier_export[carrier, node, time] != np.inf:
        return (model.export_carrier_flow[carrier, node, time] <= params.availability_carrier_export[carrier, node, time])
    else:
        return pe.Constraint.Skip


def constraint_availability_carrier_import_yearly_rule(model, carrier, node, year):
    """node- and year-dependent carrier availability to import from outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    if params.availability_carrier_import_yearly[carrier, node, year] != np.inf:
        return (params.availability_carrier_import_yearly[carrier, node, year] >= sum(
            model.import_carrier_flow[carrier, node, time] * params.time_steps_operation_duration[carrier, time] for time in EnergySystem.encode_time_step(carrier, base_time_step, yearly=True)))
    else:
        return pe.Constraint.Skip


def constraint_availability_carrier_export_yearly_rule(model, carrier, node, year):
    """node- and year-dependent carrier availability to export to outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    if params.availability_carrier_export_yearly[carrier, node, year] != np.inf:
        return (params.availability_carrier_export_yearly[carrier, node, year] >= sum(
            model.export_carrier_flow[carrier, node, time] * params.time_steps_operation_duration[carrier, time] for time in EnergySystem.encode_time_step(carrier, base_time_step, yearly=True)))
    else:
        return pe.Constraint.Skip


def constraint_cost_carrier_rule(model, carrier, node, time):
    """ cost of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    if params.availability_carrier_import[carrier, node, time] != 0 or params.availability_carrier_export[carrier, node, time] != 0:
        return (model.cost_carrier[carrier, node, time] == params.import_price_carrier[carrier, node, time] * model.import_carrier_flow[carrier, node, time] - params.export_price_carrier[
            carrier, node, time] * model.export_carrier_flow[carrier, node, time])
    else:
        return (model.cost_carrier[carrier, node, time] == 0)


def constraint_cost_shed_demand_rule(model, carrier, node, time):
    """ cost of shedding demand of carrier """
    # get parameter object
    params = Parameter.get_component_object()
    if params.shed_demand_price[carrier] != np.inf:
        return (model.cost_shed_demand_carrier[carrier, node, time] == model.shed_demand_carrier[carrier, node, time] * params.shed_demand_price[carrier])
    else:
        return (model.shed_demand_carrier[carrier, node, time] == 0)

def constraint_cost_shed_demand_high_rule(model, carrier, node, time):
    """ cost of shedding demand of carrier at high price """
    # get parameter object
    params = Parameter.get_component_object()
    if params.shed_demand_price_high[carrier] != np.inf:
        return (model.cost_shed_demand_carrier_high[carrier, node, time] == model.shed_demand_carrier_high[carrier, node, time] * params.shed_demand_price_high[carrier])
    else:
        return (model.shed_demand_carrier_high[carrier, node, time] == 0)


def constraint_limit_shed_demand_rule(model, carrier, node, time):
    """ limit demand shedding at low price """
    # get parameter object
    params = Parameter.get_component_object()
    if params.max_shed_demand[carrier] < 1:
        return (model.shed_demand_carrier[carrier, node, time] <= params.demand_carrier[carrier, node, time] * params.max_shed_demand[carrier])
    else:
        return (pe.Constraint.Skip)

def constraint_limit_shed_demand_high_rule(model, carrier, node, time):
    """ limit demand shedding at high price """
    # get parameter object
    params = Parameter.get_component_object()
    if params.max_shed_demand_high[carrier] < 1:
        return (model.shed_demand_carrier_high[carrier, node, time] <= params.demand_carrier[carrier, node, time] * params.max_shed_demand_high[carrier])
    else:
        return (pe.Constraint.Skip)


def constraint_cost_carrier_total_rule(model, year):
    """ total cost of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    return (model.cost_carrier_total[year] ==
            sum(
                sum(
                    (model.cost_carrier[carrier, node, time]
                     + model.cost_shed_demand_carrier[carrier, node, time]
                     + model.cost_shed_demand_carrier_high[carrier, node, time])
                    * params.time_steps_operation_duration[carrier, time] for time in
        EnergySystem.encode_time_step(carrier, base_time_step, yearly=True)) for carrier, node in Element.create_custom_set(["set_carriers", "set_nodes"])[0]))


def constraint_carbon_emissions_carrier_rule(model, carrier, node, time):
    """ carbon emissions of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(carrier, time)
    yearly_time_step = EnergySystem.encode_time_step(None, base_time_step, "yearly")
    if params.availability_carrier_import[carrier, node, time] != 0 or params.availability_carrier_export[carrier, node, time] != 0:
        return (model.carbon_emissions_carrier[carrier, node, time] == params.carbon_intensity_carrier[carrier, node, yearly_time_step] * (
                    model.import_carrier_flow[carrier, node, time] - model.export_carrier_flow[carrier, node, time]))
    else:
        return (model.carbon_emissions_carrier[carrier, node, time] == 0)

def constraint_carbon_emissions_carrier_total_rule(model, year):
    """ total carbon emissions of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    return (model.carbon_emissions_carrier_total[year] == sum(
        sum(model.carbon_emissions_carrier[carrier, node, time] * params.time_steps_operation_duration[carrier, time] for time in EnergySystem.encode_time_step(carrier, base_time_step, yearly=True))
        for carrier, node in Element.create_custom_set(["set_carriers", "set_nodes"])[0]))


def constraint_nodal_energy_balance_rule(model, carrier, node, time):
    """
    nodal energy balance for each time step.
    """
    # get parameter object
    params = Parameter.get_component_object()
    # carrier input and output conversion technologies
    carrier_conversion_in, carrier_conversion_out = 0, 0
    for tech in model.set_conversion_technologies:
        if carrier in model.set_input_carriers[tech]:
            carrier_conversion_in += model.input_flow[tech, carrier, node, time]
        if carrier in model.set_output_carriers[tech]:
            carrier_conversion_out += model.output_flow[tech, carrier, node, time]
    # carrier flow transport technologies
    carrier_flow_in, carrier_flow_out = 0, 0
    set_edges_in = EnergySystem.calculate_connected_edges(node, "in")
    set_edges_out = EnergySystem.calculate_connected_edges(node, "out")
    for tech in model.set_transport_technologies:
        if carrier in model.set_reference_carriers[tech]:
            carrier_flow_in += sum(model.carrier_flow[tech, edge, time] - model.carrier_loss[tech, edge, time] for edge in set_edges_in)
            carrier_flow_out += sum(model.carrier_flow[tech, edge, time] for edge in set_edges_out)
    # carrier flow storage technologies
    carrier_flow_discharge, carrier_flow_charge = 0, 0
    for tech in model.set_storage_technologies:
        if carrier in model.set_reference_carriers[tech]:
            carrier_flow_discharge += model.carrier_flow_discharge[tech, node, time]
            carrier_flow_charge += model.carrier_flow_charge[tech, node, time]
    # carrier import, demand and export
    carrier_import = model.import_carrier_flow[carrier, node, time]
    carrier_export = model.export_carrier_flow[carrier, node, time]
    carrier_demand = params.demand_carrier[carrier, node, time]
    # shed demand
    carrier_shed_demand = model.shed_demand_carrier[carrier, node, time]
    carrier_shed_demand_high = model.shed_demand_carrier_high[carrier, node, time]
    return (# conversion technologies
            carrier_conversion_out - carrier_conversion_in # transport technologies
            + carrier_flow_in - carrier_flow_out # storage technologies
            + carrier_flow_discharge - carrier_flow_charge # import and export
            + carrier_import - carrier_export # demand
            - carrier_demand # shed demand
            + carrier_shed_demand + carrier_shed_demand_high
            == 0)
