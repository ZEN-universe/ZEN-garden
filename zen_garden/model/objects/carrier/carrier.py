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
from ..element import Element
from ..energy_system import EnergySystem

class Carrier(Element):
    # set label
    label = "set_carriers"
    # empty list of elements
    list_of_elements = []

    def __init__(self, carrier: str, energy_system: EnergySystem):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model
        :param energy_system: The energy system the element is part of"""

        logging.info(f'Initialize carrier {carrier}')
        super().__init__(carrier, energy_system)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        set_base_time_steps_yearly = self.energy_system.set_base_time_steps_yearly
        set_time_steps_yearly = self.energy_system.set_time_steps_yearly
        # set attributes of carrier
        # raw import
        self.raw_time_series = {}
        self.raw_time_series["demand_carrier"] = self.data_input.extract_input_data("demand_carrier", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availability_carrier_import"] = self.data_input.extract_input_data("availability_carrier_import", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availability_carrier_export"] = self.data_input.extract_input_data("availability_carrier_export", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["export_price_carrier"] = self.data_input.extract_input_data("export_price_carrier", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["import_price_carrier"] = self.data_input.extract_input_data("import_price_carrier", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        # non-time series input data
        self.availability_carrier_import_yearly = self.data_input.extract_input_data("availability_carrier_import_yearly", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.availability_carrier_export_yearly = self.data_input.extract_input_data("availability_carrier_export_yearly", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.carbon_intensity_carrier = self.data_input.extract_input_data("carbon_intensity", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.shed_demand_price = self.data_input.extract_input_data("shed_demand_price", index_sets=[])

    def overwrite_time_steps(self, base_time_steps):
        """ overwrites set_time_steps_operation"""
        set_time_steps_operation = self.energy_system.encode_time_step(self.name, base_time_steps=base_time_steps, time_step_type="operation", yearly=True)
        setattr(self, "set_time_steps_operation", set_time_steps_operation.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_sets(cls, energy_system: EnergySystem):
        """ constructs the pe.Sets of the class <Carrier>
        :param energy_system: The Energy system to add everything"""
        pass

    @classmethod
    def construct_params(cls, energy_system: EnergySystem):
        """ constructs the pe.Params of the class <Carrier>
        :param energy_system: The Energy system to add everything"""
        # demand of carrier
        energy_system.parameters.add_parameter(name="demand_carrier", data=energy_system.initialize_component(cls, "demand_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the carrier demand')
        # availability of carrier
        energy_system.parameters.add_parameter(name="availability_carrier_import",
            data=energy_system.initialize_component(cls, "availability_carrier_import", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries')
        # availability of carrier
        energy_system.parameters.add_parameter(name="availability_carrier_export",
            data=energy_system.initialize_component(cls, "availability_carrier_export", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries')
        # availability of carrier
        energy_system.parameters.add_parameter(name="availability_carrier_import_yearly",
            data=energy_system.initialize_component(cls, "availability_carrier_import_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year')
        # availability of carrier
        energy_system.parameters.add_parameter(name="availability_carrier_export_yearly",
            data=energy_system.initialize_component(cls, "availability_carrier_export_yearly", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year')
        # import price
        energy_system.parameters.add_parameter(name="import_price_carrier", data=energy_system.initialize_component(cls, "import_price_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the import carrier price')
        # export price
        energy_system.parameters.add_parameter(name="export_price_carrier", data=energy_system.initialize_component(cls, "export_price_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_operation"]),
            doc='Parameter which specifies the export carrier price')
        # demand shedding price
        energy_system.parameters.add_parameter(name="shed_demand_price", data=energy_system.initialize_component(cls, "shed_demand_price", index_names=["set_carriers"]),
            doc='Parameter which specifies the price to shed demand')
        # carbon intensity
        energy_system.parameters.add_parameter(name="carbon_intensity_carrier",
            data=energy_system.initialize_component(cls, "carbon_intensity_carrier", index_names=["set_carriers", "set_nodes", "set_time_steps_yearly"]),
            doc='Parameter which specifies the carbon intensity of carrier')

    @classmethod
    def construct_vars(cls, energy_system: EnergySystem):
        """ constructs the pe.Vars of the class <Carrier>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model

        # flow of imported carrier
        energy_system.variables.add_variable(model, name="import_carrier_flow", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            doc='node- and time-dependent carrier import from the grid')
        # flow of exported carrier
        energy_system.variables.add_variable(model, name="export_carrier_flow", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            doc='node- and time-dependent carrier export from the grid')
        # carrier import/export cost
        energy_system.variables.add_variable(model, name="cost_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.Reals,
            doc='node- and time-dependent carrier cost due to import and export')
        # total carrier import/export cost
        energy_system.variables.add_variable(model, name="cost_carrier_total", index_sets=model.set_time_steps_yearly, domain=pe.Reals, doc='total carrier cost due to import and export')
        # carbon emissions
        energy_system.variables.add_variable(model, name="carbon_emissions_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.Reals,
            doc="carbon emissions of importing and exporting carrier")
        # carbon emissions carrier
        energy_system.variables.add_variable(model, name="carbon_emissions_carrier_total", index_sets=model.set_time_steps_yearly, domain=pe.Reals, doc="total carbon emissions of importing and exporting carrier")
        # shed demand
        energy_system.variables.add_variable(model, name="shed_demand_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            doc="shed demand of carrier")
        # cost of shed demand
        energy_system.variables.add_variable(model, name="cost_shed_demand_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            doc="shed demand of carrier")

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if np.size(energy_system.system[subclass.label]):
                subclass.construct_vars(energy_system)

    @classmethod
    def construct_constraints(cls, energy_system: EnergySystem):
        """ constructs the pe.Constraints of the class <Carrier>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model
        rules = CarrierRules(energy_system)
        # limit import flow by availability
        energy_system.constraints.add_constraint(model, name="constraint_availability_carrier_import", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_availability_carrier_import_rule, doc='node- and time-dependent carrier availability to import from outside the system boundaries', )
        # limit export flow by availability
        energy_system.constraints.add_constraint(model, name="constraint_availability_carrier_export", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_availability_carrier_export_rule, doc='node- and time-dependent carrier availability to export to outside the system boundaries', )
        # limit import flow by availability for each year
        energy_system.constraints.add_constraint(model, name="constraint_availability_carrier_import_yearly", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"], energy_system),
            rule=rules.constraint_availability_carrier_import_yearly_rule, doc='node- and time-dependent carrier availability to import from outside the system boundaries summed over entire year', )
        # limit export flow by availability for each year
        energy_system.constraints.add_constraint(model, name="constraint_availability_carrier_export_yearly", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_yearly"], energy_system),
            rule=rules.constraint_availability_carrier_export_yearly_rule, doc='node- and time-dependent carrier availability to export to outside the system boundaries summed over entire year', )
        # cost for carrier
        energy_system.constraints.add_constraint(model, name="constraint_cost_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system), rule=rules.constraint_cost_carrier_rule,
            doc="cost of importing and exporting carrier")
        # cost for shed demand
        energy_system.constraints.add_constraint(model, name="constraint_cost_shed_demand", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_cost_shed_demand_rule, doc="cost of shedding carrier demand")
        # limit of shed demand
        energy_system.constraints.add_constraint(model, name="constraint_limit_shed_demand", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_limit_shed_demand_rule, doc="limit of shedding carrier demand")
        # total cost for carriers
        energy_system.constraints.add_constraint(model, name="constraint_cost_carrier_total", index_sets=model.set_time_steps_yearly, rule=rules.constraint_cost_carrier_total_rule,
            doc="total cost of importing and exporting carriers")
        # carbon emissions
        energy_system.constraints.add_constraint(model, name="constraint_carbon_emissions_carrier", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_carbon_emissions_carrier_rule, doc="carbon emissions of importing and exporting carrier")

        # carbon emissions carrier
        energy_system.constraints.add_constraint(model, name="constraint_carbon_emissions_carrier_total", index_sets=model.set_time_steps_yearly, rule=rules.constraint_carbon_emissions_carrier_total_rule,
            doc="total carbon emissions of importing and exporting carriers")
        # energy balance
        energy_system.constraints.add_constraint(model, name="constraint_nodal_energy_balance", index_sets=cls.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            rule=rules.constraint_nodal_energy_balance_rule, doc='node- and time-dependent energy balance for each carrier', )
        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if len(energy_system.system[subclass.label]) > 0:
                subclass.construct_constraints(energy_system)


class CarrierRules:
    """
    Rules for the Carrier class
    """

    def __init__(self, energy_system: EnergySystem):
        """
        Inits the rules for a given EnergySystem
        :param energy_system: The EnergySystem
        """

        self.energy_system = energy_system

    # %% Constraint rules defined in current class
    def constraint_availability_carrier_import_rule(self, model, carrier, node, time):
        """node- and time-dependent carrier availability to import from outside the system boundaries"""
        # get parameter object
        params = self.energy_system.parameters
        if params.availability_carrier_import[carrier, node, time] != np.inf:
            return (model.import_carrier_flow[carrier, node, time] <= params.availability_carrier_import[carrier, node, time])
        else:
            return pe.Constraint.Skip

    def constraint_availability_carrier_export_rule(self, model, carrier, node, time):
        """node- and time-dependent carrier availability to export to outside the system boundaries"""
        # get parameter object
        params = self.energy_system.parameters
        if params.availability_carrier_export[carrier, node, time] != np.inf:
            return (model.export_carrier_flow[carrier, node, time] <= params.availability_carrier_export[carrier, node, time])
        else:
            return pe.Constraint.Skip

    def constraint_availability_carrier_import_yearly_rule(self, model, carrier, node, year):
        """node- and year-dependent carrier availability to import from outside the system boundaries"""
        # get parameter object
        params = self.energy_system.parameters
        base_time_step = self.energy_system.decode_time_step(None, year, "yearly")
        if params.availability_carrier_import_yearly[carrier, node, year] != np.inf:
            return (params.availability_carrier_import_yearly[carrier, node, year] >= sum(
                model.import_carrier_flow[carrier, node, time] * params.time_steps_operation_duration[carrier, time] for time in self.energy_system.encode_time_step(carrier, base_time_step, yearly=True)))
        else:
            return pe.Constraint.Skip

    def constraint_availability_carrier_export_yearly_rule(self, model, carrier, node, year):
        """node- and year-dependent carrier availability to export to outside the system boundaries"""
        # get parameter object
        params = self.energy_system.parameters
        base_time_step = self.energy_system.decode_time_step(None, year, "yearly")
        if params.availability_carrier_export_yearly[carrier, node, year] != np.inf:
            return (params.availability_carrier_export_yearly[carrier, node, year] >= sum(
                model.export_carrier_flow[carrier, node, time] * params.time_steps_operation_duration[carrier, time] for time in self.energy_system.encode_time_step(carrier, base_time_step, yearly=True)))
        else:
            return pe.Constraint.Skip

    def constraint_cost_carrier_rule(self, model, carrier, node, time):
        """ cost of importing and exporting carrier"""
        # get parameter object
        params = self.energy_system.parameters
        return (model.cost_carrier[carrier, node, time] == params.import_price_carrier[carrier, node, time] * model.import_carrier_flow[carrier, node, time] - params.export_price_carrier[
            carrier, node, time] * model.export_carrier_flow[carrier, node, time])

    def constraint_cost_shed_demand_rule(self, model, carrier, node, time):
        """ cost of shedding demand of carrier """
        # get parameter object
        params = self.energy_system.parameters
        if params.shed_demand_price[carrier] != np.inf:
            return (model.cost_shed_demand_carrier[carrier, node, time] == model.shed_demand_carrier[carrier, node, time] * params.shed_demand_price[carrier])
        else:
            return (model.shed_demand_carrier[carrier, node, time] == 0)

    def constraint_limit_shed_demand_rule(self, model, carrier, node, time):
        """ limit of shedding demand of carrier to demand --> endogenous demand cannot not be supplied by shed demand """
        # get parameter object
        params = self.energy_system.parameters
        return (model.shed_demand_carrier[carrier, node, time] <= params.demand_carrier[carrier, node, time])

    def constraint_cost_carrier_total_rule(self, model, year):
        """ total cost of importing and exporting carrier"""
        # get parameter object
        params = self.energy_system.parameters
        base_time_step = self.energy_system.decode_time_step(None, year, "yearly")
        return (model.cost_carrier_total[year] == sum(sum(
            (model.cost_carrier[carrier, node, time] + model.cost_shed_demand_carrier[carrier, node, time]) * params.time_steps_operation_duration[carrier, time] for time in
            self.energy_system.encode_time_step(carrier, base_time_step, yearly=True)) for carrier, node in Element.create_custom_set(["set_carriers", "set_nodes"], self.energy_system)[0]))

    def constraint_carbon_emissions_carrier_rule(self, model, carrier, node, time):
        """ carbon emissions of importing and exporting carrier"""
        # get parameter object
        params = self.energy_system.parameters
        base_time_step = self.energy_system.decode_time_step(carrier, time)
        yearly_time_step = self.energy_system.encode_time_step(None, base_time_step, "yearly")
        return (model.carbon_emissions_carrier[carrier, node, time] == params.carbon_intensity_carrier[carrier, node, yearly_time_step] * (
                    model.import_carrier_flow[carrier, node, time] - model.export_carrier_flow[carrier, node, time]))

    def constraint_carbon_emissions_carrier_total_rule(self, model, year):
        """ total carbon emissions of importing and exporting carrier"""
        # get parameter object
        params = self.energy_system.parameters
        base_time_step = self.energy_system.decode_time_step(None, year, "yearly")
        return (model.carbon_emissions_carrier_total[year] == sum(
            sum(model.carbon_emissions_carrier[carrier, node, time] * params.time_steps_operation_duration[carrier, time] for time in self.energy_system.encode_time_step(carrier, base_time_step, yearly=True))
            for carrier, node in Element.create_custom_set(["set_carriers", "set_nodes"], self.energy_system)[0]))

    def constraint_nodal_energy_balance_rule(self, model, carrier, node, time):
        """
        nodal energy balance for each time step.
        """
        # get parameter object
        params = self.energy_system.parameters
        # carrier input and output conversion technologies
        carrier_conversion_in, carrier_conversion_out = 0, 0
        for tech in model.set_conversion_technologies:
            if carrier in model.set_input_carriers[tech]:
                carrier_conversion_in += model.input_flow[tech, carrier, node, time]
            if carrier in model.set_output_carriers[tech]:
                carrier_conversion_out += model.output_flow[tech, carrier, node, time]
        # carrier flow transport technologies
        carrier_flow_in, carrier_flow_out = 0, 0
        set_edges_in = self.energy_system.calculate_connected_edges(node, "in")
        set_edges_out = self.energy_system.calculate_connected_edges(node, "out")
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
        return (# conversion technologies
                carrier_conversion_out - carrier_conversion_in # transport technologies
                + carrier_flow_in - carrier_flow_out # storage technologies
                + carrier_flow_discharge - carrier_flow_charge # import and export
                + carrier_import - carrier_export # demand
                - carrier_demand # shed demand
                + carrier_shed_demand == 0)
