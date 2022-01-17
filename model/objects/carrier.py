"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a generic energy carrier.
                The class takes as inputs the abstract optimization model. The class adds parameters, variables and
                constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.objects.element import Element
# from preprocess.functions.calculate_input_data import extractInputData

class Carrier(Element):
    # empty list of elements
    listOfElements = []

    def __init__(self, object,carrier):
        """initialization of a generic carrier object
        :param object: object of the abstract optimization model
        :param carrier: carrier that is added to the model"""

        logging.info('initialize object of a generic carrier')
        super().__init__(object,carrier)
        # add carrier to list
        Carrier.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get system information
        system = Element.getSystem()   
        paths = Element.getPaths()   
        indexNames = Element.getAnalysis()['dataInputs']
        # set attributes of carrier
        # sets
        
        # parameters
        if self.name in system["setImportCarriers"]:
            _inputPath = paths["setImportCarriers"][self.name]["folder"]
            self.availabilityCarrier = Element.extractInputData(_inputPath,"availabilityCarrier",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
            self.exportPriceCarrier = Element.extractInputData(_inputPath,"exportPriceCarrier",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
            self.importPriceCarrier = Element.extractInputData(_inputPath,"importPriceCarrier",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
        elif self.name in system["setExportCarriers"]:
            _inputPath = paths["setExportCarriers"][self.name]["folder"]
            self.demandCarrier = Element.extractInputData(_inputPath,"demandCarrier",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
            self.exportPriceCarrier = Element.extractInputData(_inputPath,"exportPriceCarrier",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
            self.importPriceCarrier = Element.extractInputData(_inputPath,"importPriceCarrier",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])

    ### --- classmethods to define sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def defineSets(cls):
        """ defines the pe.Sets of the class <Carrier> """
        model = cls.getConcreteModel()
        
        # Import carriers
        model.setImportCarriers = pe.Set(
            initialize = Element.getAttributeOfElement("grid","setImportCarriers"),
            doc='Set of technology specific Import carriers. Import defines the import over the system boundaries.')
        # Export carriers
        model.setExportCarriers = pe.Set(
            initialize = Element.getAttributeOfElement("grid","setExportCarriers"),
            doc='Set of technology specific Export carriers. Export defines the export over the system boundaries.')
        
    @classmethod
    def defineParams(cls):
        """ defines the pe.Params of the class <Carrier> """
        model = cls.getConcreteModel()

        # demand of carrier
        model.demandCarrier = pe.Param(
            model.setExportCarriers,
            model.setNodes,
            model.setTimeSteps,
            initialize = cls.getAttributeOfAllElements("demandCarrier"),
            doc = 'Parameter which specifies the carrier demand.\n\t Dimensions: setExportCarriers, setNodes, setTimeSteps')
        # availability of carrier
        model.availabilityCarrier = pe.Param(
            model.setImportCarriers,
            model.setNodes,
            model.setTimeSteps,
            initialize = cls.getAttributeOfAllElements("availabilityCarrier"),
            doc = 'Parameter which specifies the maximum energy that can be imported from the grid. \n\t Dimensions: setImportCarriers, setNodes, setTimeSteps')
        # import price
        model.importPriceCarrier = pe.Param(
            model.setCarriers,
            model.setNodes,
            model.setTimeSteps,
            initialize = cls.getAttributeOfAllElements("importPriceCarrier"),
            doc = 'Parameter which specifies the import carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeSteps'
        )
        # export price
        model.exportPriceCarrier = pe.Param(
            model.setCarriers,
            model.setNodes,
            model.setTimeSteps,
            initialize = cls.getAttributeOfAllElements("exportPriceCarrier"),
            doc = 'Parameter which specifies the export carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeSteps'
        )

    @classmethod
    def defineVars(cls):
        """ defines the pe.Vars of the class <Carrier> """
        model = cls.getConcreteModel()
        
        # flow of imported carrier
        model.importCarrierFlow = pe.Var(
            model.setImportCarriers,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier import from the grid. \n\t Dimensions: setImportCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'
        )
        # flow of exported carrier
        model.exportCarrierFlow = pe.Var(
            model.setExportCarriers,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier export from the grid. \n\t Dimensions: setExportCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'
        )

    @classmethod
    def defineConstraints(cls):
        """ defines the pe.Constraints of the class <Carrier> """
        model = cls.getConcreteModel()

        # limit import flow by availability
        model.constraintAvailabilityCarrier = pe.Constraint(
            model.setImportCarriers,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintAvailabilityCarrierRule,
            doc = 'node- and time-dependent carrier availability. \n\t Dimensions: setImportCarriers, setNodes, setTimeSteps',
        )        
        ### TODO add mass balance but move after technologies
        # energy balance
        model.constraintNodalEnergyBalance = pe.Constraint(
            model.setCarriers,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintNodalEnergyBalanceRule,
            doc = 'node- and time-dependent energy balance for each carrier. \n\t Dimensions: setCarriers, setNodes, setTimeSteps',
        )

#%% Constraint rules defined in current class
def constraintAvailabilityCarrierRule(model, carrier, node, time):
    """node- and time-dependent carrier availability"""

    return(model.importCarrierFlow[carrier, node, time] <= model.availabilityCarrier[carrier,node,time])

# energy balance
def constraintNodalEnergyBalanceRule(model, carrier, node, time):
    """" 
    nodal energy balance for each time step
    """

    # carrier input and output conversion technologies
    carrierConversionIn, carrierConversionOut = 0, 0
    if hasattr(model, 'setConversionTechnologies'):
        for tech in model.setConversionTechnologies:
            if carrier in model.setInputCarriers[tech]:
                carrierConversionIn += model.inputFlow[tech,carrier,node,time]
            if carrier in model.setOutputCarriers[tech]:
                carrierConversionOut += model.outputFlow[tech,carrier,node,time]
    # carrier flow transport technologies
    carrierFlowIn, carrierFlowOut = 0, 0
    if hasattr(model, 'setTransportTechnologies'):
        for tech in model.setTransportTechnologies:
            if carrier in model.setTransportCarriers[tech]:
                carrierFlowIn += sum(model.carrierFlow[tech,carrier, edge, time]
                                        - model.carrierLoss[tech,carrier, edge, time] for edge in model.setEdges if node == model.setNodesOnEdges[edge][2]) # second entry is node into which the flow goes
                carrierFlowOut += sum(model.carrierFlow[tech,carrier, edge, time] for edge in model.setEdges if node == model.setNodesOnEdges[edge][1]) # first entry is node out of which the flow starts
    # carrier import, demand and export
    carrierImport, carrierExport, carrierDemand = 0, 0, 0
    if carrier in model.setImportCarriers:
        carrierImport = model.importCarrierFlow[carrier, node, time]
    if carrier in model.setExportCarriers:
        carrierDemand = model.demandCarrier[carrier, node, time]
        carrierExport = model.exportCarrierFlow[carrier, node, time]

    # TODO implement storage

    return (
        carrierConversionOut - carrierConversionIn + carrierFlowIn - carrierFlowOut + carrierImport - carrierExport - carrierDemand == 0
        )