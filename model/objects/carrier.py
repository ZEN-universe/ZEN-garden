"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining a generic energy carrier.
              The class takes as inputs the abstract optimization model. The class adds parameters, variables and
              constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.objects.element import Element

class Carrier(Element):
    # empty list of elements
    listOfElements = []

    def __init__(self, object,carrier):
        """initialization of a generic carrier object
        :param object: object of the abstract optimization model
        :param carrier: carrier that is added to the model"""

        logging.info('initialize object of a generic carrier')
        super().__init__(object,carrier)

        # set attributes of carrier
        if carrier in self.system["setImportCarriers"]:
            self.availabilityCarrier = object.pyoDict["availabilityCarrier"][carrier]
        elif carrier in self.system["setExportCarriers"]:
            self.demandCarrier = object.pyoDict["demandCarrier"][carrier]
        self.exportPriceCarrier = object.pyoDict["exportPriceCarrier"][carrier]
        self.importPriceCarrier = object.pyoDict["importPriceCarrier"][carrier]
        # add carrier to list
        Carrier.addElement(self)


        #%% Sets and subsets
        # subsets = {
        #     'setInputCarriers':      'Set of technology specific input carriers. \
        #                               \n\t Dimension: setCarriers',
        #     'setExportCarriers':     'Set of technology specific Export carriers. \
        #                               \n\t Subset: setCarriers'}
        # # self.addSets(subsets)

        # #%% Parameters
        # params = {
        #     'demandCarrier':                    'Parameter which specifies the carrier demand. \
        #                                         \n\t Dimensions: setExportCarriers, setNodes, setTimeSteps',
        #     'availabilityCarrier':              'Parameter which specifies the maximum energy that can be imported from the grid.\
        #                                         \n\t Dimensions: setInputCarriers, setNodes, setTimeSteps'
        #     # 'exportPriceCarrier': 'Parameter which specifies the export carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeSteps',
        #     # 'importPriceCarrier': 'Parameter which specifies the import carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeSteps',               
        #     # 'footprintCarrier': 'Parameter which specifies the carbon intensity of a carrier. \n\t Dimensions: setCarriers',s
        #     }
        # # self.addParams(params)

        # #%% Variables
        # variables = {
        #     'importCarrier':                    'node- and time-dependent carrier import from the grid.\
        #                                         \n\t Dimensions: setInputCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
        #     'exportCarrier':                    'node- and time-dependent carrier export from the grid. \
        #                                         \n\t Dimensions: setExportCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'}
        # # self.addVars(variables)

        # #%% Contraints in current class
        # constr = {
        #     'AvailabilityCarrier':    'node- and time-dependent carrier availability.\
        #                                \n\t Dimensions: setInputCarriers, setNodes, setTimeSteps',
        #     }
        # # self.addConstr(constr)
        
        # logging.info('added carrier sets, parameters, decision variables and constraints')

    ### --- classmethods --- ###
    # setter/getter classmethods
    

    ### --- classmethods to define sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def defineSets(cls):
        """ defines the pe.Sets of the class <Carrier> """
        model = cls.getConcreteModel()
        pyoDict = cls.getPyoDict()
        
        # Import carriers
        model.setImportCarriers = pe.Set(
            initialize = pyoDict["setImportCarriers"],
            doc='Set of technology specific Import carriers. Import defines the import over the system boundaries.')
        # Export carriers
        model.setExportCarriers = pe.Set(
            initialize = pyoDict["setExportCarriers"],
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
#%% Constraint rules defined in current class
def constraintAvailabilityCarrierRule(model, carrier, node, time):
    """node- and time-dependent carrier availability"""

    return(model.importCarrierFlow[carrier, node, time] <= model.availabilityCarrier[carrier,node,time])