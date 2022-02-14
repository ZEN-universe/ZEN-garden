"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all transport technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for the transport technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.objects.technology.technology import Technology
from model.objects.energy_system import EnergySystem
from preprocess.functions.time_series_aggregation import TimeSeriesAggregation

class TransportTechnology(Technology):
    # empty list of elements
    listOfElements = []
    
    def __init__(self, tech):
        """init transport technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize transport technology {tech}')
        super().__init__(tech)
        # store input data
        self.storeInputData()
        # add TransportTechnology to list
        TransportTechnology.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get attributes from class <Technology>
        super().storeInputData()
        # get system information
        paths               = EnergySystem.getPaths()   
        setBaseTimeSteps    = EnergySystem.getEnergySystem().setBaseTimeSteps
        # set attributes for parameters of parent class <Technology>
        _inputPath          = paths["setTransportTechnologies"][self.name]["folder"]
        # add all raw time series to dict
        self.rawTimeSeries                  = {}
        self.rawTimeSeries["minLoad"]       = self.dataInput.extractInputData(_inputPath,"minLoad",indexSets=["setEdges","setTimeSteps"],timeSteps=setBaseTimeSteps,transportTechnology=True)
        self.rawTimeSeries["maxLoad"]       = self.dataInput.extractInputData(_inputPath,"maxLoad",indexSets=["setEdges","setTimeSteps"],timeSteps=setBaseTimeSteps,transportTechnology=True)
        self.rawTimeSeries["opexSpecific"]  = self.dataInput.extractInputData(_inputPath,"opexSpecific",indexSets=["setEdges","setTimeSteps"],timeSteps= setBaseTimeSteps,transportTechnology=True)
        # non-time series input data
        self.capacityLimit                  = self.dataInput.extractInputData(_inputPath,"capacityLimit",indexSets=["setEdges"],transportTechnology=True)
        self.carbonIntensityTechnology      = self.dataInput.extractInputData(_inputPath,"carbonIntensity",indexSets=["setEdges"])
        # set attributes for parameters of child class <TransportTechnology>
        # TODO calculate for non Euclidean distance
        self.distance                       = self.dataInput.extractInputData(_inputPath,"distanceEuclidean",indexSets=["setEdges"],transportTechnology=True)
        self.capexPerDistance               = self.dataInput.extractInputData(_inputPath,"capexPerDistance",indexSets=["setEdges","setTimeSteps"],timeSteps= self.setTimeStepsInvest,transportTechnology=True)
        self.lossFlow                       = self.dataInput.extractAttributeData(_inputPath,"lossFlow")
        # set technology to correspondent reference carrier
        EnergySystem.setTechnologyOfCarrier(self.name,self.referenceCarrier)
        # apply time series aggregation
        TimeSeriesAggregation(self,_inputPath)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to TransportTechnology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <TransportTechnology> """
        pass

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <TransportTechnology> """
        model = EnergySystem.getConcreteModel()
        
        # distance between nodes
        model.distance = pe.Param(
            cls.createCustomSet(["setTransportTechnologies","setEdges"]),
            initialize = cls.getAttributeOfAllElements("distance"),
            doc = 'distance between two nodes for transport technologies. Dimensions: setTransportTechnologies, setEdges')
        # cost per distance
        model.capexPerDistance = pe.Param(
            cls.createCustomSet(["setTransportTechnologies","setEdges","setTimeStepsInvest"]),
            initialize = cls.getAttributeOfAllElements("capexPerDistance"),
            doc = 'capex per unit distance for transport technologies. Dimensions: setTransportTechnologies, setEdges, setTimeStepsInvest')
        # carrier losses
        model.lossFlow = pe.Param(
            model.setTransportTechnologies,
            initialize = cls.getAttributeOfAllElements("lossFlow"),
            doc = 'carrier losses due to transport with transport technologies. Dimensions: setTransportTechnologies')

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <TransportTechnology> """
        def carrierFlowBounds(model,tech ,edge,time):
            """ return bounds of carrierFlow for bigM expression 
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param edge: edge index
            :param time: time index
            :return bounds: bounds of carrierFlow"""
            # convert operationTimeStep to investTimeStep: operationTimeStep -> baseTimeStep -> investTimeStep
            baseTimeStep = EnergySystem.decodeTimeStep(tech,time,"operation")
            investTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"invest")
            bounds = model.capacity[tech,edge,investTimeStep].bounds
            return(bounds)

        model = EnergySystem.getConcreteModel()
        # flow of carrier on edge
        model.carrierFlow = pe.Var(
            cls.createCustomSet(["setTransportTechnologies","setEdges","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'carrier flow through transport technology on edge i and time t. Dimensions: setTransportTechnologies, setEdges, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        # loss of carrier on edge
        model.carrierLoss = pe.Var(
            cls.createCustomSet(["setTransportTechnologies","setEdges","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'carrier flow through transport technology on edge i and time t. Dimensions: setTransportTechnologies, setEdges, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        
    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <TransportTechnology> """
        model = EnergySystem.getConcreteModel()

        # Carrier Flow Losses 
        model.constraintTransportTechnologyLossesFlow = pe.Constraint(
            cls.createCustomSet(["setTransportTechnologies","setEdges","setTimeStepsOperation"]),
            rule = constraintTransportTechnologyLossesFlowRule,
            doc = 'Carrier loss due to transport with through transport technology. Dimensions: setTransportTechnologies, setEdges, setTimeStepsOperation'
        ) 
        # Linear Capex
        model.constraintTransportTechnologyLinearCapex = pe.Constraint(
            cls.createCustomSet(["setTransportTechnologies","setEdges","setTimeStepsInvest"]),
            rule = constraintCapexTransportTechnologyRule,
            doc = 'Capital expenditures for installing transport technology. Dimensions: setTransportTechnologies, setEdges, setTimeStepsInvest'
        ) 

    # defines disjuncts if technology on/off
    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech, edge, time):
        """definition of disjunct constraints if technology is on"""
        model = disjunct.model()
        # get invest time step
        baseTimeStep = EnergySystem.decodeTimeStep(tech,time,"operation")
        investTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"invest")
        # disjunct constraints min load
        disjunct.constraintMinLoad = pe.Constraint(
            expr=model.carrierFlow[tech, edge, time] >= model.minLoad[tech,edge,time] * model.capacity[tech,edge, investTimeStep]
        )

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, edge, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraintNoLoad = pe.Constraint(
            expr=model.carrierFlow[tech, edge, time] == 0
        )

### --- functions with constraint rules --- ###
def constraintTransportTechnologyLossesFlowRule(model, tech, edge, time):
    """compute the flow losses for a carrier through a transport technology"""
    return(model.carrierLoss[tech, edge, time]
            == model.distance[tech,edge] * model.lossFlow[tech] * model.carrierFlow[tech, edge, time])

def constraintCapexTransportTechnologyRule(model, tech, edge, time):
    """ definition of the capital expenditures for the transport technology"""
    return (model.capex[tech,edge, time] == 
            model.builtCapacity[tech,edge, time] *
            model.distance[tech,edge] *
            model.capexPerDistance[tech,edge, time])
