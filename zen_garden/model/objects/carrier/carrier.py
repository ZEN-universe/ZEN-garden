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
from ..component import Parameter,Variable,Constraint

class Carrier(Element):
    # set label
    label = "setCarriers"
    # empty list of elements
    list_of_elements = []

    def __init__(self,carrier):
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
        set_base_time_steps_yearly          = EnergySystem.get_energy_system().set_base_time_steps_yearly
        set_time_steps_yearly              = EnergySystem.get_energy_system().set_time_steps_yearly
        # set attributes of carrier
        # raw import
        self.raw_time_series                              = {}
        self.raw_time_series["demandCarrier"]             = self.datainput.extract_input_data("demandCarrier",index_sets = ["setNodes","set_time_steps"],time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availabilityCarrierImport"] = self.datainput.extract_input_data("availabilityCarrier",index_sets = ["setNodes","set_time_steps"],column="availabilityCarrierImport",time_steps=set_base_time_steps_yearly)
        self.raw_time_series["availabilityCarrierExport"] = self.datainput.extract_input_data("availabilityCarrier",index_sets = ["setNodes","set_time_steps"],column="availabilityCarrierExport",time_steps=set_base_time_steps_yearly)
        self.raw_time_series["exportPriceCarrier"]        = self.datainput.extract_input_data("priceCarrier",index_sets = ["setNodes","set_time_steps"],column="exportPriceCarrier",time_steps=set_base_time_steps_yearly)
        self.raw_time_series["importPriceCarrier"]        = self.datainput.extract_input_data("priceCarrier",index_sets = ["setNodes","set_time_steps"],column="importPriceCarrier",time_steps=set_base_time_steps_yearly)
        # non-time series input data
        self.availabilityCarrierImportYearly            = self.datainput.extract_input_data("availabilityCarrierYearly",index_sets = ["setNodes","set_time_steps"],column="availabilityCarrierImportYearly",time_steps=set_time_steps_yearly)
        self.availabilityCarrierExportYearly            = self.datainput.extract_input_data("availabilityCarrierYearly",index_sets = ["setNodes","set_time_steps"],column="availabilityCarrierExportYearly",time_steps=set_time_steps_yearly)
        self.carbonIntensityCarrier                     = self.datainput.extract_input_data("carbonIntensity",index_sets = ["setNodes","set_time_steps"],time_steps=set_time_steps_yearly)
        self.shedDemandPrice                            = self.datainput.extract_input_data("shedDemandPrice",index_sets = [])
        
    def overwrite_time_steps(self,base_time_steps):
        """ overwrites setTimeStepsOperation and  setTimeStepsEnergyBalance"""
        setTimeStepsOperation       = EnergySystem.encode_time_step(self.name, base_time_steps=base_time_steps, time_step_type="operation",yearly=True)
        setattr(self, "setTimeStepsOperation", setTimeStepsOperation.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <Carrier> """
        pass

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <Carrier> """
        # demand of carrier
        Parameter.add_parameter(
            name="demandCarrier",
            data= EnergySystem.initialize_component(cls,"demandCarrier",index_names=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the carrier demand')
        # availability of carrier
        Parameter.add_parameter(
            name="availabilityCarrierImport",
            data= EnergySystem.initialize_component(cls,"availabilityCarrierImport",index_names=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the maximum energy that can be imported from outside the system boundaries')
        # availability of carrier
        Parameter.add_parameter(
            name="availabilityCarrierExport",
            data= EnergySystem.initialize_component(cls,"availabilityCarrierExport",index_names=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the maximum energy that can be exported to outside the system boundaries')
        # availability of carrier
        Parameter.add_parameter(
            name="availabilityCarrierImportYearly",
            data= EnergySystem.initialize_component(cls,"availabilityCarrierImportYearly",index_names=["setCarriers","setNodes","set_time_steps_yearly"]),
            doc = 'Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year')
        # availability of carrier
        Parameter.add_parameter(
            name="availabilityCarrierExportYearly",
            data= EnergySystem.initialize_component(cls,"availabilityCarrierExportYearly",index_names=["setCarriers","setNodes","set_time_steps_yearly"]),
            doc = 'Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year')
        # import price
        Parameter.add_parameter(
            name="importPriceCarrier",
            data= EnergySystem.initialize_component(cls,"importPriceCarrier",index_names=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the import carrier price')
        # export price
        Parameter.add_parameter(
            name="exportPriceCarrier",
            data= EnergySystem.initialize_component(cls,"exportPriceCarrier",index_names=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the export carrier price')
        # demand shedding price
        Parameter.add_parameter(
            name="shedDemandPrice",
            data=EnergySystem.initialize_component(cls, "shedDemandPrice",index_names=["setCarriers"]),
            doc='Parameter which specifies the price to shed demand')
        # carbon intensity
        Parameter.add_parameter(
            name="carbonIntensityCarrier",
            data= EnergySystem.initialize_component(cls,"carbonIntensityCarrier",index_names=["setCarriers","setNodes","set_time_steps_yearly"]),
            doc = 'Parameter which specifies the carbon intensity of carrier')

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <Carrier> """
        model = EnergySystem.get_pyomo_model()
        
        # flow of imported carrier
        Variable.add_variable(
            model,
            name="importCarrierFlow",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier import from the grid'
        )
        # flow of exported carrier
        Variable.add_variable(
            model,
            name="exportCarrierFlow",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier export from the grid'
        )
        # carrier import/export cost
        Variable.add_variable(
            model,
            name="costCarrier",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = 'node- and time-dependent carrier cost due to import and export'
        )
        # total carrier import/export cost
        Variable.add_variable(
            model,
            name="cost_carrier_total",
            index_sets= model.set_time_steps_yearly,
            domain = pe.Reals,
            doc = 'total carrier cost due to import and export'
        )
        # carbon emissions
        Variable.add_variable(
            model,
            name="carbonEmissionsCarrier",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = "carbon emissions of importing and exporting carrier"
        )
        # carbon emissions carrier
        Variable.add_variable(
            model,
            name="carbon_emissions_carrier_total",
            index_sets= model.set_time_steps_yearly,
            domain=pe.Reals,
            doc="total carbon emissions of importing and exporting carrier"
        )
        # shed demand
        Variable.add_variable(
            model,
            name="shedDemandCarrier",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain=pe.NonNegativeReals,
            doc="shed demand of carrier"
        )
        # cost of shed demand
        Variable.add_variable(
            model,
            name="costShedDemandCarrier",
            index_sets= cls.create_custom_set(["setCarriers", "setNodes", "setTimeStepsOperation"]),
            domain=pe.NonNegativeReals,
            doc="shed demand of carrier"
        )

        # add pe.Sets of the child classes
        for subclass in cls.get_all_subclasses():
            if np.size(EnergySystem.get_system()[subclass.label]):
                subclass.construct_vars()

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.get_pyomo_model()

        # limit import flow by availability
        Constraint.add_constraint(
            model,
            name="constraintAvailabilityCarrierImport",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintAvailabilityCarrierImportRule,
            doc = 'node- and time-dependent carrier availability to import from outside the system boundaries',
        )        
        # limit export flow by availability
        Constraint.add_constraint(
            model,
            name="constraintAvailabilityCarrierExport",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintAvailabilityCarrierExportRule,
            doc = 'node- and time-dependent carrier availability to export to outside the system boundaries',
        )
        # limit import flow by availability for each year
        Constraint.add_constraint(
            model,
            name="constraintAvailabilityCarrierImportYearly",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","set_time_steps_yearly"]),
            rule = constraintAvailabilityCarrierImportYearlyRule,
            doc = 'node- and time-dependent carrier availability to import from outside the system boundaries summed over entire year',
        )
        # limit export flow by availability for each year
        Constraint.add_constraint(
            model,
            name="constraintAvailabilityCarrierExportYearly",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","set_time_steps_yearly"]),
            rule = constraintAvailabilityCarrierExportYearlyRule,
            doc = 'node- and time-dependent carrier availability to export to outside the system boundaries summed over entire year',
        )
        # cost for carrier
        Constraint.add_constraint(
            model,
            name="constraintCostCarrier",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintCostCarrierRule,
            doc = "cost of importing and exporting carrier"
        )
        # cost for carrier
        Constraint.add_constraint(
            model,
            name="constraintCostShedDemand",
            index_sets= cls.create_custom_set(["setCarriers", "setNodes", "setTimeStepsOperation"]),
            rule=constraintCostShedDemandRule,
            doc="cost of shedding carrier demand"
        )
        # total cost for carriers
        Constraint.add_constraint(
            model,
            name="constraintCostCarrierTotal",
            index_sets= model.set_time_steps_yearly,
            rule = constraintCostCarrierTotalRule,
            doc = "total cost of importing and exporting carriers"
        )
        # carbon emissions
        Constraint.add_constraint(
            model,
            name="constraintCarbonEmissionsCarrier",
            index_sets= cls.create_custom_set(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintCarbonEmissionsCarrierRule,
            doc = "carbon emissions of importing and exporting carrier"
        )

        # carbon emissions carrier
        Constraint.add_constraint(
            model,
            name="constraintCarbonEmissionsCarrierTotal",
            index_sets= model.set_time_steps_yearly,
            rule=constraintCarbonEmissionsCarrierTotalRule,
            doc="total carbon emissions of importing and exporting carriers"
        )
        # energy balance
        Constraint.add_constraint(
            model,
            name="constraintNodalEnergyBalance",
            index_sets= cls.create_custom_set(["setCarriers", "setNodes", "setTimeStepsOperation"]),
            rule=constraintNodalEnergyBalanceRule,
            doc='node- and time-dependent energy balance for each carrier',
        )
        # add pe.Sets of the child classes
        for subclass in cls.get_all_subclasses():
            if np.size(EnergySystem.get_system()[subclass.label]):
                subclass.construct_constraints()


#%% Constraint rules defined in current class
def constraintAvailabilityCarrierImportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to import from outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    if params.availabilityCarrierImport[carrier,node,time] != np.inf:
        return(model.importCarrierFlow[carrier, node, time] <= params.availabilityCarrierImport[carrier,node,time])
    else:
        return pe.Constraint.Skip

def constraintAvailabilityCarrierExportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to export to outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    if params.availabilityCarrierExport[carrier,node,time] != np.inf:
        return(model.exportCarrierFlow[carrier, node, time] <= params.availabilityCarrierExport[carrier,node,time])
    else:
        return pe.Constraint.Skip

def constraintAvailabilityCarrierImportYearlyRule(model, carrier, node, year):
    """node- and year-dependent carrier availability to import from outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    if params.availabilityCarrierImportYearly[carrier,node,year] != np.inf:
        return(
            params.availabilityCarrierImportYearly[carrier, node, year] >=
            sum(
                model.importCarrierFlow[carrier, node, time]
                * params.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encode_time_step(carrier, base_time_step, yearly=True)
                )
        )
    else:
        return pe.Constraint.Skip

def constraintAvailabilityCarrierExportYearlyRule(model, carrier, node, year):
    """node- and year-dependent carrier availability to export to outside the system boundaries"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    if params.availabilityCarrierExportYearly[carrier,node,year] != np.inf:
        return (
            params.availabilityCarrierExportYearly[carrier, node, year] >=
            sum(
                model.exportCarrierFlow[carrier, node, time]
                * params.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encode_time_step(carrier, base_time_step, yearly=True)
            )
        )
    else:
        return pe.Constraint.Skip

def constraintCostCarrierRule(model, carrier, node, time):
    """ cost of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    return(model.costCarrier[carrier,node, time] ==
        params.importPriceCarrier[carrier, node, time]*model.importCarrierFlow[carrier, node, time] -
        params.exportPriceCarrier[carrier, node, time]*model.exportCarrierFlow[carrier, node, time]
    )

def constraintCostShedDemandRule(model, carrier, node, time):
    """ cost of shedding demand of carrier """
    # get parameter object
    params = Parameter.get_component_object()
    if params.shedDemandPrice[carrier] != np.inf:
        return(
            model.costShedDemandCarrier[carrier,node, time] ==
            model.shedDemandCarrier[carrier,node,time] * params.shedDemandPrice[carrier]
        )
    else:
        return(
            model.shedDemandCarrier[carrier, node, time] == 0
        )

def constraintCostCarrierTotalRule(model,year):
    """ total cost of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    return(model.cost_carrier_total[year] ==
        sum(
            sum(
                (model.costCarrier[carrier,node,time] + model.costShedDemandCarrier[carrier,node, time])
                * params.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encode_time_step(carrier, base_time_step, yearly=True)
            )
            for carrier,node in Element.create_custom_set(["setCarriers","setNodes"])[0]
        )
    )

def constraintCarbonEmissionsCarrierRule(model, carrier, node, time):
    """ carbon emissions of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step    = EnergySystem.decode_time_step(carrier, time)
    yearlyTimeStep  = EnergySystem.encode_time_step(None,base_time_step,"yearly")
    return (model.carbonEmissionsCarrier[carrier, node, time] ==
            params.carbonIntensityCarrier[carrier, node, yearlyTimeStep] *
            (model.importCarrierFlow[carrier, node, time] - model.exportCarrierFlow[carrier, node, time])
            )

def constraintCarbonEmissionsCarrierTotalRule(model, year):
    """ total carbon emissions of importing and exporting carrier"""
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None,year,"yearly")
    return(model.carbon_emissions_carrier_total[year] ==
        sum(
            sum(
                model.carbonEmissionsCarrier[carrier, node, time] * params.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encode_time_step(carrier, base_time_step, yearly = True)
            )
            for carrier, node in Element.create_custom_set(["setCarriers", "setNodes"])[0]
        )
    )

def constraintNodalEnergyBalanceRule(model, carrier, node, time):
    """" 
    nodal energy balance for each time step. 
    The constraint is indexed by setTimeStepsOperation, which is union of time step sequences of all corresponding technologies and carriers
    timeStepEnergyBalance --> base_time_step --> element_time_step
    """
    # get parameter object
    params = Parameter.get_component_object()
    # carrier input and output conversion technologies
    carrierConversionIn, carrierConversionOut = 0, 0
    for tech in model.setConversionTechnologies:
        if carrier in model.setInputCarriers[tech]:
            carrierConversionIn     += model.inputFlow[tech,carrier,node,time]
        if carrier in model.setOutputCarriers[tech]:
            carrierConversionOut    += model.outputFlow[tech,carrier,node,time]
    # carrier flow transport technologies
    carrierFlowIn, carrierFlowOut   = 0, 0
    setEdgesIn                      = EnergySystem.calculate_connected_edges(node,"in")
    setEdgesOut                     = EnergySystem.calculate_connected_edges(node,"out")
    for tech in model.setTransportTechnologies:
        if carrier in model.setReferenceCarriers[tech]:
            carrierFlowIn   += sum(model.carrierFlow[tech, edge, time]
                            - model.carrierLoss[tech, edge, time] for edge in setEdgesIn)
            carrierFlowOut  += sum(model.carrierFlow[tech, edge, time] for edge in setEdgesOut)
    # carrier flow storage technologies
    carrierFlowDischarge, carrierFlowCharge = 0, 0
    for tech in model.setStorageTechnologies:
        if carrier in model.setReferenceCarriers[tech]:
            carrierFlowDischarge    += model.carrierFlowDischarge[tech,node,time]
            carrierFlowCharge       += model.carrierFlowCharge[tech,node,time]
    # carrier import, demand and export
    carrierImport, carrierExport, carrierDemand = 0, 0, 0
    carrierImport       = model.importCarrierFlow[carrier, node, time]
    carrierExport       = model.exportCarrierFlow[carrier, node, time]
    carrierDemand       = params.demandCarrier[carrier, node, time]
    # shed demand
    carrierShedDemand   = model.shedDemandCarrier[carrier, node, time]
    return (
        # conversion technologies
        carrierConversionOut - carrierConversionIn
        # transport technologies
        + carrierFlowIn - carrierFlowOut
        # storage technologies
        + carrierFlowDischarge - carrierFlowCharge
        # import and export
        + carrierImport - carrierExport
        # demand
        - carrierDemand
        # shed demand
        + carrierShedDemand
        == 0
    )
