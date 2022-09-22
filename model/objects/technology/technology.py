"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for all technologies.
==========================================================================================================================================================================="""
import copy
import logging
import pyomo.environ as pe
import pyomo.gdp as pgdp
import numpy as np
import cProfile
import pstats
from model.objects.element import Element
from model.objects.energy_system import EnergySystem
from model.objects.parameter import Parameter

class Technology(Element):
    # set label
    label           = "setTechnologies"
    locationType    = None
    # empty list of elements
    listOfElements = []

    def __init__(self, technology):
        """init generic technology object
        :param object: object of the abstract optimization model
        :param technology: technology that is added to the model"""

        super().__init__(technology)
        # add Technology to list
        Technology.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # set attributes of technology
        _setLocation    = type(self).getClassLocationType()

        setBaseTimeStepsYearly          = EnergySystem.getEnergySystem().setBaseTimeStepsYearly
        self.setTimeStepsInvest         = self.dataInput.extractTimeSteps(self.name,typeOfTimeSteps="invest")
        self.setTimeStepsInvestEntireHorizon    = copy.deepcopy(self.setTimeStepsInvest)

        self.timeStepsInvestDuration    = EnergySystem.calculateTimeStepDuration(self.setTimeStepsInvest)
        self.sequenceTimeStepsInvest    = np.concatenate([[timeStep]*self.timeStepsInvestDuration[timeStep] for timeStep in self.timeStepsInvestDuration])
        EnergySystem.setSequenceTimeSteps(self.name,self.sequenceTimeStepsInvest,timeStepType="invest")
        self.referenceCarrier           = [self.dataInput.extractAttributeData("referenceCarrier",skipWarning=True)]
        EnergySystem.setTechnologyOfCarrier(self.name, self.referenceCarrier)
        self.minBuiltCapacity           = self.dataInput.extractAttributeData("minBuiltCapacity")["value"]
        self.maxBuiltCapacity           = self.dataInput.extractAttributeData("maxBuiltCapacity")["value"]
        self.lifetime                   = self.dataInput.extractAttributeData("lifetime")["value"]
        self.constructionTime           = self.dataInput.extractAttributeData("constructionTime")["value"]
        # maximum diffusion rate
        self.maxDiffusionRate           = self.dataInput.extractInputData("maxDiffusionRate", indexSets=["setTimeSteps"],timeSteps=self.setTimeStepsInvest)

        # add all raw time series to dict
        self.rawTimeSeries = {}
        self.rawTimeSeries["minLoad"]       = self.dataInput.extractInputData("minLoad",indexSets=[_setLocation, "setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["maxLoad"]       = self.dataInput.extractInputData("maxLoad",indexSets=[_setLocation, "setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["opexSpecific"]  = self.dataInput.extractInputData("opexSpecific",indexSets=[_setLocation,"setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        # non-time series input data
        self.fixedOpexSpecific          = self.dataInput.extractInputData("fixedOpexSpecific",indexSets=[_setLocation,"setTimeSteps"],timeSteps=self.setTimeStepsInvest)
        self.capacityLimit              = self.dataInput.extractInputData("capacityLimit",indexSets=[_setLocation])
        self.carbonIntensityTechnology  = self.dataInput.extractInputData("carbonIntensity",indexSets=[_setLocation])
        # extract existing capacity
        self.setExistingTechnologies    = self.dataInput.extractSetExistingTechnologies()
        self.existingCapacity           = self.dataInput.extractInputData("existingCapacity",indexSets=[_setLocation,"setExistingTechnologies"])
        self.existingInvestedCapacity   = self.dataInput.extractInputData("existingInvestedCapacity", indexSets=[_setLocation, "setTimeSteps"], timeSteps=EnergySystem.getEnergySystem().setTimeStepsYearly)
        self.lifetimeExistingTechnology = self.dataInput.extractLifetimeExistingTechnology("existingCapacity",indexSets=[_setLocation,"setExistingTechnologies"])

    def calculateCapexOfExistingCapacities(self,storageEnergy = False):
        """ this method calculates the annualized capex of the existing capacities """
        if storageEnergy:
            existingCapacities  = self.existingCapacityEnergy
        else:
            existingCapacities  = self.existingCapacity
        if self.__class__.__name__ == "StorageTechnology":
            existingCapex   = existingCapacities.to_frame().apply(
                lambda _existingCapacity: self.calculateCapexOfSingleCapacity(_existingCapacity.squeeze(),_existingCapacity.name,storageEnergy),axis=1)
        else:
            existingCapex   = existingCapacities.to_frame().apply(
                lambda _existingCapacity: self.calculateCapexOfSingleCapacity(_existingCapacity.squeeze(),_existingCapacity.name), axis=1)
        return existingCapex

    def calculateCapexOfSingleCapacity(self,*args):
        """ this method calculates the annualized capex of the existing capacities. Is implemented in child class """
        raise NotImplementedError

    def calculateFractionalAnnuity(self):
        """calculate fraction of annuity to depreciate investment"""
        system              = EnergySystem.getSystem()
        _lifetime           = self.lifetime
        _annuity            = 1/_lifetime
        # only account for fraction of year
        _fractionOfYear     = system["unaggregatedTimeStepsPerYear"] / system["totalHoursPerYear"]
        _fractionalAnnuity  = _annuity * _fractionOfYear
        return _fractionalAnnuity

    def overwriteTimeSteps(self,baseTimeSteps):
        """ overwrites setTimeStepsInvest and setTimeStepsOperation """
        setTimeStepsInvest      = EnergySystem.encodeTimeStep(self.name,baseTimeSteps=baseTimeSteps,timeStepType="invest",yearly=True)
        setTimeStepsOperation   = EnergySystem.encodeTimeStep(self.name, baseTimeSteps=baseTimeSteps,timeStepType="operation",yearly=True)

        # copy invest time steps
        self.setTimeStepsInvest                 = setTimeStepsInvest.squeeze().tolist()
        self.setTimeStepsOperation              = setTimeStepsOperation.squeeze().tolist()

    def addNewlyBuiltCapacityTech(self,builtCapacity,capex,baseTimeSteps):
        """ adds the newly built capacity to the existing capacity
        :param builtCapacity: pd.Series of newly built capacity of technology
        :param capex: pd.Series of capex of newly built capacity of technology
        :param baseTimeSteps: base time steps of current horizon step """
        system = EnergySystem.getSystem()
        # reduce lifetime of existing capacities and add new remaining lifetime
        self.lifetimeExistingTechnology             = (self.lifetimeExistingTechnology - system["intervalBetweenYears"]).clip(lower=0)
        # new capacity
        _investTimeSteps                            = EnergySystem.encodeTimeStep(self.name, baseTimeSteps, "invest", yearly=True)
        # TODO currently summed over all invest time steps, fix for #TS_investPerYear >1
        _newlyBuiltCapacity                         = builtCapacity[_investTimeSteps].sum(axis=1)
        _capex                                      = capex[_investTimeSteps].sum(axis=1)
        # if at least one value unequal to zero
        if not (_newlyBuiltCapacity == 0).all():
            # add new index to setExistingTechnologies
            indexNewTechnology                          = max(self.setExistingTechnologies) + 1
            self.setExistingTechnologies                = np.append(self.setExistingTechnologies, indexNewTechnology)
            # add new remaining lifetime
            _lifetimeTechnology                         = self.lifetimeExistingTechnology.unstack()
            _lifetimeTechnology[indexNewTechnology]     = self.lifetime
            self.lifetimeExistingTechnology             = _lifetimeTechnology.stack()

            for typeCapacity in list(set(_newlyBuiltCapacity.index.get_level_values(0))):
                # if power
                if typeCapacity == system["setCapacityTypes"][0]:
                    _energyString = ""
                # if energy
                else:
                    _energyString = "Energy"
                _existingCapacity       = getattr(self,"existingCapacity"+_energyString)
                _capexExistingCapacity  = getattr(self, "capexExistingCapacity" + _energyString)
                # add new existing capacity
                _existingCapacity                           = _existingCapacity.unstack()
                _existingCapacity[indexNewTechnology]       = _newlyBuiltCapacity.loc[typeCapacity]
                setattr(self,"existingCapacity"+_energyString,_existingCapacity.stack())
                # calculate capex of existing capacity
                _capexExistingCapacity                      = _capexExistingCapacity.unstack()
                _capexExistingCapacity[indexNewTechnology]  = _capex.loc[typeCapacity]
                setattr(self, "capexExistingCapacity" + _energyString,_capexExistingCapacity.stack())

    def addNewlyInvestedCapacityTech(self,investedCapacity,stepHorizon):
        """ adds the newly invested capacity to the list of invested capacity
        :param investedCapacity: pd.Series of newly built capacity of technology
        :param stepHorizon: optimization time step """
        system = EnergySystem.getSystem()
        _newlyInvestedCapacity = investedCapacity[stepHorizon]
        _newlyInvestedCapacity = _newlyInvestedCapacity.fillna(0)
        if not (_newlyInvestedCapacity == 0).all():
            for typeCapacity in list(set(_newlyInvestedCapacity.index.get_level_values(0))):
                # if power
                if typeCapacity == system["setCapacityTypes"][0]:
                    _energyString = ""
                # if energy
                else:
                    _energyString = "Energy"
                _existingInvestedCapacity = getattr(self, "existingInvestedCapacity" + _energyString)
                # add new existing invested capacity
                _existingInvestedCapacity = _existingInvestedCapacity.unstack()
                _existingInvestedCapacity[stepHorizon] = _newlyInvestedCapacity.loc[typeCapacity]
                setattr(self, "existingInvestedCapacity" + _energyString, _existingInvestedCapacity.stack())

    ### --- getter/setter classmethods
    @classmethod
    def getClassLocationType(cls):
        """ returns locationType of class """
        return cls.locationType

    ### --- classmethods
    @classmethod
    def getLifetimeRange(cls, tech, time, timeStepType: str = None):
        """ returns lifetime range of technology. If timeStepType, then converts the yearly time step 'time' to timeStepType """
        if timeStepType:
            baseTimeSteps   = EnergySystem.decodeTimeStep(None, time, "yearly")
            investTimeStep  = EnergySystem.encodeTimeStep(tech, baseTimeSteps, timeStepType, yearly=True)
        else:
            investTimeStep  = time
        tStart, tEnd = cls.getStartEndTimeOfPeriod(tech, investTimeStep)

        return range(tStart, tEnd + 1)

    @classmethod
    def getAvailableExistingQuantity(cls, tech,capacityType,loc, time,typeExistingQuantity, timeStepType: str = None):
        """ returns existing quantity of 'tech', that is still available at invest time step 'time'.
        Either capacity or capex.
        :param tech: name of technology
        :param loc: location (node or edge) of existing capacity
        :param time: current time
        :param idExistingCapacity: id of existing capacity
        :return existingQuantity: existing capacity or capex of existing capacity
        """
        params = Parameter.getParameterObject()
        system = EnergySystem.getSystem()
        discountRate = EnergySystem.getAnalysis()["discountRate"]
        if timeStepType:
            baseTimeSteps   = EnergySystem.decodeTimeStep(None, time, "yearly")
            investTimeStep  = EnergySystem.encodeTimeStep(tech, baseTimeSteps, timeStepType, yearly=True)
        else:
            investTimeStep  = time

        model               = EnergySystem.getConcreteModel()
        existingQuantity = 0
        if typeExistingQuantity == "capacity":
            existingVariable = params.existingCapacity
        elif typeExistingQuantity == "capex":
            existingVariable = params.capexExistingCapacity
        else:
            raise KeyError(f"Wrong type of existing quantity {typeExistingQuantity}")

        for idExistingCapacity in model.setExistingTechnologies[tech]:
            tStart  = cls.getStartEndTimeOfPeriod(tech, investTimeStep, idExistingCapacity=idExistingCapacity,loc= loc)
            # discount existing capex
            if typeExistingQuantity == "capex":
                yearConstruction = max(0,time*system["intervalBetweenYears"] - params.lifetimeTechnology[tech] + params.lifetimeExistingTechnology[tech,loc,idExistingCapacity])
                discountFactor = (1 + discountRate)**(time*system["intervalBetweenYears"] - yearConstruction)
            else:
                discountFactor = 1
            # if still available at first base time step, add to list
            if tStart == model.setBaseTimeSteps.at(1):
                existingQuantity += existingVariable[tech,capacityType, loc, idExistingCapacity]*discountFactor
        return existingQuantity

    @classmethod
    def getStartEndTimeOfPeriod(cls, tech, investTimeStep,periodType = "lifetime",clipToFirstTimeStep = True, idExistingCapacity = None,loc = None):
        """ counts back the period (either lifetime of constructionTime) back to get the start invest time step and returns startInvestTimeStep
        :param tech: name of technology
        :param investTimeStep: current investment time step
        :param periodType: "lifetime" if lifetime is counted backwards, "constructionTime" if construction time is counted backwards
        :param clipToFirstTimeStep: boolean to clip the time step to first time step if time step too far in the past
        :param idExistingCapacity: id of existing capacity
        :param loc: location (node or edge) of existing capacity
        :return beganInPast: boolean if the period began before the first optimization step
        :return startInvestTimeStep,endInvestTimeStep: start and end of period in invest time step domain"""

        # get model and system
        params  = Parameter.getParameterObject()
        model   = EnergySystem.getConcreteModel()
        system  = EnergySystem.getSystem()
        # get which period to count backwards
        if periodType == "lifetime":
            periodTime = params.lifetimeTechnology
        elif periodType == "constructionTime":
            periodTime = params.constructionTimeTechnology
        else:
            raise NotImplemented(f"getStartEndOfPeriod not yet implemented for {periodType}")
        # get endInvestTimeStep
        if not isinstance(investTimeStep, np.ndarray):
            endInvestTimeStep = investTimeStep
        elif len(investTimeStep) == 1:
            endInvestTimeStep = investTimeStep[0]
        # if more than one investment time step
        else:
            endInvestTimeStep = investTimeStep[-1]
            investTimeStep = investTimeStep[0]
        # convert period to interval of base time steps
        if idExistingCapacity is None:
            periodYearly = periodTime[tech]
        else:
            assert periodType == "lifetime", "Existing planned capacities not yet implemented"
            periodYearly = params.lifetimeExistingTechnology[tech, loc, idExistingCapacity]
        basePeriod = periodYearly / system["intervalBetweenYears"] * system["unaggregatedTimeStepsPerYear"]
        basePeriod = round(basePeriod, EnergySystem.getSolver()["roundingDecimalPoints"])
        if int(basePeriod) != basePeriod:
            logging.warning(
                f"The period {periodType} of {tech} does not translate to an integer time interval in the base time domain ({basePeriod})")
        # decode to base time steps
        baseTimeSteps = EnergySystem.decodeTimeStep(tech, investTimeStep, timeStepType="invest")
        if len(baseTimeSteps) == 0:
            return model.setBaseTimeSteps.at(1), model.setBaseTimeSteps.at(1) - 1
        baseTimeStep = baseTimeSteps[0]

        # if startBaseTimeStep is further in the past than first base time step, use first base time step
        if clipToFirstTimeStep:
            startBaseTimeStep   = int(max(model.setBaseTimeSteps.at(1), baseTimeStep - basePeriod + 1))
        else:
            startBaseTimeStep   = int(baseTimeStep - basePeriod + 1)
        # if period of existing capacity, then only return the start base time step
        if idExistingCapacity is not None:
            return startBaseTimeStep
        startInvestTimeStep     = EnergySystem.encodeTimeStep(tech, startBaseTimeStep, timeStepType="invest", yearly=True)[0]

        return startInvestTimeStep, endInvestTimeStep

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Technology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <Technology> """
        # construct the pe.Sets of the class <Technology>
        model = EnergySystem.getConcreteModel()

        # conversion technologies
        model.setConversionTechnologies = pe.Set(
            initialize=EnergySystem.getAttribute("setConversionTechnologies"),
            doc='Set of conversion technologies. Subset: setTechnologies')
        # transport technologies
        model.setTransportTechnologies = pe.Set(
            initialize=EnergySystem.getAttribute("setTransportTechnologies"),
            doc='Set of transport technologies. Subset: setTechnologies')
        # storage technologies
        model.setStorageTechnologies = pe.Set(
            initialize=EnergySystem.getAttribute("setStorageTechnologies"),
            doc='Set of storage technologies. Subset: setTechnologies')
        # existing installed technologies
        model.setExistingTechnologies = pe.Set(
            model.setTechnologies,
            initialize=cls.getAttributeOfAllElements("setExistingTechnologies"),
            doc='Set of existing technologies. Subset: setTechnologies')
        # invest time steps
        model.setTimeStepsInvest = pe.Set(
            model.setTechnologies,
            initialize = cls.getAttributeOfAllElements("setTimeStepsInvest"),
            doc="Set of time steps in investment for all technologies. Dimensions: setTechnologies"
        )
        # invest time steps
        model.setTimeStepsInvestEntireHorizon = pe.Set(
            model.setTechnologies,
            initialize=cls.getAttributeOfAllElements("setTimeStepsInvestEntireHorizon"),
            doc="Set of time steps in investment for all technologies of entire horizon. Dimensions: setTechnologies"
        )
        # reference carriers
        model.setReferenceCarriers = pe.Set(
            model.setTechnologies,
            initialize = cls.getAttributeOfAllElements("referenceCarrier"),
            doc = "set of all reference carriers correspondent to a technology. Dimensions: setTechnologies"
        )
        # add pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructSets()

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <Technology> """
        # construct pe.Param of the class <Technology>

        # invest time step duration
        Parameter.addParameter(
            name="timeStepsInvestDuration",
            data= EnergySystem.initializeComponent(cls,"timeStepsInvestDuration",indexNames=["setTechnologies","setTimeStepsInvest"]).astype(int),
            doc="Parameter which specifies the time step duration in investment for all technologies. Dimensions: setTechnologies, setTimeStepsInvest")
        # existing capacity
        Parameter.addParameter(
            name="existingCapacity",
            data=EnergySystem.initializeComponent(cls,"existingCapacity",indexNames=["setTechnologies","setCapacityTypes", "setLocation", "setExistingTechnologies"],capacityTypes=True),
            doc='Parameter which specifies the existing technology size. Dimensions: setTechnologies')
        # existing capacity
        Parameter.addParameter(
            name="existingInvestedCapacity",
            data=EnergySystem.initializeComponent(cls, "existingInvestedCapacity",indexNames=["setTechnologies", "setCapacityTypes","setLocation", "setTimeStepsInvestEntireHorizon"],capacityTypes=True),
            doc='Parameter which specifies the size of the previously invested capacities. ''Dimensions: setTechnologies, setCapacityTypes, setLocation, setTimeStepsInvestEntireHorizon')
        # minimum capacity
        Parameter.addParameter(
            name="minBuiltCapacity",
            data= EnergySystem.initializeComponent(cls,"minBuiltCapacity",capacityTypes=True),
            doc = 'Parameter which specifies the minimum technology size that can be installed. Dimensions: setTechnologies')
        # maximum capacity
        Parameter.addParameter(
            name="maxBuiltCapacity",
            data= EnergySystem.initializeComponent(cls,"maxBuiltCapacity",capacityTypes=True),
            doc = 'Parameter which specifies the maximum technology size that can be installed. Dimensions: setTechnologies')
        # lifetime existing technologies
        Parameter.addParameter(
            name="lifetimeExistingTechnology",
            data=EnergySystem.initializeComponent(cls,"lifetimeExistingTechnology"),
            doc='Parameter which specifies the remaining lifetime of an existing technology. Dimensions: setTechnologies, setLocation, setExistingTechnologies')
        # lifetime existing technologies
        Parameter.addParameter(
            name="capexExistingCapacity",
            data=EnergySystem.initializeComponent(cls,"capexExistingCapacity",capacityTypes=True),
            doc='Parameter which specifies the annualized capex of an existing technology which still has to be paid. Dimensions: setTechnologies')
        # lifetime newly built technologies
        Parameter.addParameter(
            name="lifetimeTechnology",
            data= EnergySystem.initializeComponent(cls,"lifetime"),
            doc = 'Parameter which specifies the lifetime of a newly built technology. Dimensions: setTechnologies')
        # constructionTime newly built technologies
        Parameter.addParameter(
            name="constructionTimeTechnology",
            data=EnergySystem.initializeComponent(cls, "constructionTime"),
            doc='Parameter which specifies the construction time of a newly built technology. Dimensions: setTechnologies')
        # maximum diffusion rate, i.e., increase in capacity
        Parameter.addParameter(
            name="maxDiffusionRate",
            data=EnergySystem.initializeComponent(cls, "maxDiffusionRate",indexNames=["setTechnologies", "setTimeStepsInvest"]),
            doc="Parameter which specifies the maximum diffusion rate, i.e., the maximum increase in capacity between investment steps. Dimensions: setTechnologies, setTimeStepsInvest"
        )
        # capacityLimit of technologies
        Parameter.addParameter(
            name="capacityLimitTechnology",
            data= EnergySystem.initializeComponent(cls,"capacityLimit",capacityTypes=True),
            doc = 'Parameter which specifies the capacity limit of technologies. Dimensions: setTechnologies, setLocation')
        # minimum load relative to capacity
        Parameter.addParameter(
            name="minLoad",
            data= EnergySystem.initializeComponent(cls,"minLoad",indexNames=["setTechnologies","setCapacityTypes","setLocation","setTimeStepsOperation"],capacityTypes=True),
            doc = 'Parameter which specifies the minimum load of technology relative to installed capacity. Dimensions:setTechnologies, setLocation, setTimeStepsOperation')
        # maximum load relative to capacity
        Parameter.addParameter(
            name="maxLoad",
            data= EnergySystem.initializeComponent(cls,"maxLoad",indexNames=["setTechnologies","setCapacityTypes","setLocation","setTimeStepsOperation"],capacityTypes=True),
            doc = 'Parameter which specifies the maximum load of technology relative to installed capacity. Dimensions:setTechnologies, setLocation, setTimeStepsOperation')
        # specific opex
        Parameter.addParameter(
            name="opexSpecific",
            data= EnergySystem.initializeComponent(cls,"opexSpecific",indexNames=["setTechnologies","setLocation","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the specific opex. Dimensions: setTechnologies, setLocation, setTimeStepsOperation')
        # carbon intensity
        Parameter.addParameter(
            name="carbonIntensityTechnology",
            data= EnergySystem.initializeComponent(cls,"carbonIntensityTechnology"),
            doc = 'Parameter which specifies the carbon intensity of each technology. Dimensions: setTechnologies, setLocation')
        # add pe.Param of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructParams()

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <Technology> """
        def capacityBounds(model,tech,capacityType, loc, time):
            """ return bounds of capacity for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param capacityType: either power or energy
            :param loc: location of capacity
            :param time: investment time step
            :return bounds: bounds of capacity"""
            # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
            # or if transportTechnology and enforceSelfishBehavior
            system = EnergySystem.getSystem()
            if tech in techsOnOff or ("enforceSelfishBehavior" in system.keys() and system["enforceSelfishBehavior"] and tech in system["setTransportTechnologies"]):
                params = Parameter.getParameterObject()
                if capacityType == system["setCapacityTypes"][0]:
                    _energyString = ""
                else:
                    _energyString = "Energy"
                _existingCapacity           = getattr(params,"existingCapacity"+_energyString)
                _maxBuiltCapacity           = getattr(params,"maxBuiltCapacity" + _energyString)
                _capacityLimitTechnology    = getattr(params,"capacityLimitTechnology" + _energyString)
                existingCapacities = 0
                for idExistingTechnology in model.setExistingTechnologies[tech]:
                    if (time - params.lifetimeExistingTechnology[tech, loc, idExistingTechnology] + 1) <= 0:
                        existingCapacities  += _existingCapacity[tech, capacityType, loc, idExistingTechnology]

                maxBuiltCapacity            = len(model.setTimeStepsInvest[tech])*_maxBuiltCapacity[tech,capacityType]
                maxCapacityLimitTechnology  = _capacityLimitTechnology[tech,capacityType, loc]
                boundCapacity = min(maxBuiltCapacity + existingCapacities,maxCapacityLimitTechnology + existingCapacities)
                bounds = (0,boundCapacity)
                return(bounds)
            else:
                return(None,None)

        model       = EnergySystem.getConcreteModel()
        # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
        techsOnOff  = Technology.createCustomSet(["setTechnologies","setOnOff"])
        # construct pe.Vars of the class <Technology>
        # install technology
        model.installTechnology = pe.Var(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            domain = pe.Binary,
            doc = 'installment of a technology at location l and time t. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest. Domain: Binary')
        # capacity technology
        model.capacity = pe.Var(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            domain = pe.NonNegativeReals,
            bounds = capacityBounds,
            doc = 'size of installed technology at location l and time t. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest. Domain: NonNegativeReals')
        # builtCapacity technology
        model.builtCapacity = pe.Var(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            domain = pe.NonNegativeReals,
            doc = 'size of built technology (invested capacity after construction) at location l and time t. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest. Domain: NonNegativeReals')
        # investedCapacity technology
        model.investedCapacity = pe.Var(
            cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsInvest"]),
            domain=pe.NonNegativeReals,
            doc='size of invested technology at location l and time t. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest. Domain: NonNegativeReals')
        # capex of building capacity
        model.capex = pe.Var(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            domain = pe.NonNegativeReals,
            doc = 'capex for building technology at location l and time t. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest. Domain: NonNegativeReals')
        # annual capex of having capacity
        model.capexYearly = pe.Var(
            cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsYearly"]),
            domain=pe.NonNegativeReals,
            doc='annual capex for having technology at location l. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsYearly. Domain: NonNegativeReals')
        # total capex
        model.capexTotal = pe.Var(
            model.setTimeStepsYearly,
            domain = pe.NonNegativeReals,
            doc = 'total capex for installing all technologies in all locations at all times. Domain: NonNegativeReals')
        # opex
        model.opex = pe.Var(
            cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = "opex for operating technology at location l and time t. Dimensions: setTechnologies, setLocation, setTimeStepsOperation. Domain: NonNegativeReals"
        )
        # total opex
        model.opexTotal = pe.Var(
            model.setTimeStepsYearly,
            domain = pe.NonNegativeReals,
            doc = "total opex for operating technology at location l and time t. Dimensions: setTechnologies, setLocation, setTimeStepsOperation. Domain: NonNegativeReals"
        )
        # carbon emissions
        model.carbonEmissionsTechnology = pe.Var(
            cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = "carbon emissions for operating technology at location l and time t. Dimensions: setTechnologies, setLocation, setTimeStepsOperation. Domain: NonNegativeReals"
        )
        # total carbon emissions technology
        model.carbonEmissionsTechnologyTotal = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="total carbon emissions for operating technology at location l and time t. Domain: NonNegativeReals"
        )

        # add pe.Vars of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructVars()

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Technology> """
        model = EnergySystem.getConcreteModel()
        # construct pe.Constraints of the class <Technology>
        #  technology capacityLimit
        model.constraintTechnologyCapacityLimit = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            rule = constraintTechnologyCapacityLimitRule,
            doc = 'limited capacity of  technology depending on loc and time. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest'
        )
        # minimum capacity
        model.constraintTechnologyMinCapacity = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            rule = constraintTechnologyMinCapacityRule,
            doc = 'min capacity of technology that can be installed. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest'
        )
        # maximum capacity
        model.constraintTechnologyMaxCapacity = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            rule = constraintTechnologyMaxCapacityRule,
            doc = 'max capacity of technology that can be installed. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest'
        )
        # construction period
        model.constraintTechnologyConstructionTime = pe.Constraint(
            cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsInvest"]),
            rule=constraintTechnologyConstructionTimeRule,
            doc='lead time in which invested technology is constructed. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest'
        )
        # lifetime
        model.constraintTechnologyLifetime = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsInvest"]),
            rule = constraintTechnologyLifetimeRule,
            doc = 'max capacity of  technology that can be installed. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsInvest'
        )
        # limit diffusion rate
        model.constraintTechnologyDiffusionLimit = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation", "setTimeStepsInvest"]),
            rule=constraintTechnologyDiffusionLimitRule,
            doc="Limits the newly built capacity by the existing knowledge stock. Dimension: setConversionTechnologies,setCapacityTypes,setLocation,setTimeStepsInvest.")
        # limit max load by installed capacity
        model.constraintMaxLoad = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsOperation"]),
            rule = constraintMaxLoadRule,
            doc = 'limit max load by installed capacity. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsOperation'
        )
        # annual capex of having capacity
        model.constraintCapexYearly = pe.Constraint(
            cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsYearly"]),
            rule=constraintCapexYearlyRule,
            doc='annual capex of having capacity of technology. Dimensions: setTechnologies,setCapacityTypes, setLocation, setTimeStepsYearly.'
        )
        # total capex of all technologies
        model.constraintCapexTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule = constraintCapexTotalRule,
            doc = 'total capex of all technology that can be installed.'
        )
        # calculate opex
        model.constraintOpexTechnology = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            rule = constraintOpexTechnologyRule,
            doc = "opex for each technology at each location and time step"
        )
        # total opex of all technologies
        model.constraintOpexTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule = constraintOpexTotalRule,
            doc = 'total opex of all technology that are operated.'
        )
        # carbon emissions of technologies
        model.constraintCarbonEmissionsTechnology = pe.Constraint(
            cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            rule = constraintCarbonEmissionsTechnologyRule,
            doc = "carbon emissions for each technology at each location and time step"
        )
        # total carbon emissions of technologies
        model.constraintCarbonEmissionsTechnologyTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsTechnologyTotalRule,
            doc="total carbon emissions for each technology at each location and time step"
        )

        # disjunct if technology is on
        model.disjunctOnTechnology = pgdp.Disjunct(
            cls.createCustomSet(["setTechnologies","setOnOff","setLocation","setTimeStepsOperation"]),
            rule = cls.disjunctOnTechnologyRule,
            doc = "disjunct to indicate that technology is On. Dimensions: setTechnologies, setLocation, setTimeStepsOperation"
        )
        # disjunct if technology is off
        model.disjunctOffTechnology = pgdp.Disjunct(
            cls.createCustomSet(["setTechnologies","setOnOff","setLocation","setTimeStepsOperation"]),
            rule = cls.disjunctOffTechnologyRule,
            doc = "disjunct to indicate that technology is off. Dimensions: setTechnologies, setLocation, setTimeStepsOperation"
        )
        # disjunction
        model.disjunctionDecisionOnOffTechnology = pgdp.Disjunction(
            cls.createCustomSet(["setTechnologies","setOnOff","setLocation","setTimeStepsOperation"]),
            rule = cls.expressionLinkDisjunctsRule,
            doc = "disjunction to link the on off disjuncts. Dimensions: setTechnologyLocation, setTimeStep")

        # add pe.Constraints of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructConstraints()

    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech, loc, time):
        """definition of disjunct constraints if technology is On
        iterate through all subclasses to find corresponding implementation of disjunct constraints """
        for subclass in cls.getAllSubclasses():
            if tech in subclass.getAllNamesOfElements():
                # disjunct is defined in corresponding subclass
                subclass.disjunctOnTechnologyRule(disjunct,tech,loc,time)
                break

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, loc, time):
        """definition of disjunct constraints if technology is off
        iterate through all subclasses to find corresponding implementation of disjunct constraints """
        for subclass in cls.getAllSubclasses():
            if tech in subclass.getAllNamesOfElements():
                # disjunct is defined in corresponding subclass
                subclass.disjunctOffTechnologyRule(disjunct,tech,loc,time)
                break

    @classmethod
    def expressionLinkDisjunctsRule(cls,model, tech, loc, time):
        """ link disjuncts for technology is on and technology is off """
        return ([model.disjunctOnTechnology[tech,loc,time],model.disjunctOffTechnology[tech,loc,time]])

### --- constraint rules --- ###
#%% Constraint rules pre-defined in Technology class
def constraintTechnologyCapacityLimitRule(model, tech,capacityType, loc, time):
    """limited capacityLimit of technology"""
    # get parameter object
    params = Parameter.getParameterObject()
    if params.capacityLimitTechnology[tech,capacityType, loc] != np.inf:
        existingCapacities = Technology.getAvailableExistingQuantity(tech, capacityType, loc, time,typeExistingQuantity="capacity")
        if existingCapacities < params.capacityLimitTechnology[tech, capacityType, loc]:
            return (params.capacityLimitTechnology[tech,capacityType, loc] >= model.capacity[tech,capacityType, loc, time])
        else:
            return (model.builtCapacity[tech, capacityType, loc, time] == 0)
    else:
        return pe.Constraint.Skip

def constraintTechnologyMinCapacityRule(model, tech,capacityType, loc, time):
    """ min capacity expansion of technology."""
    # get parameter object
    params = Parameter.getParameterObject()
    if params.minBuiltCapacity[tech,capacityType] != 0:
        return (params.minBuiltCapacity[tech,capacityType] * model.installTechnology[tech,capacityType, loc, time] <= model.builtCapacity[tech,capacityType, loc, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyMaxCapacityRule(model, tech,capacityType, loc, time):
    """max capacity expansion of technology"""
    # get parameter object
    params = Parameter.getParameterObject()
    system = EnergySystem.getSystem()
    if params.maxBuiltCapacity[tech,capacityType] != np.inf:
        return (params.maxBuiltCapacity[tech,capacityType] * model.installTechnology[tech,capacityType, loc, time] >= model.builtCapacity[tech,capacityType, loc, time])
    elif system['DoubleCapexTransport'] and tech in system["setTransportTechnologies"] and model.maxCapacity[tech,capacityType] != np.inf:
        return (params.maxCapacity[tech, capacityType] * model.installTechnology[tech, capacityType, loc, time] >= model.builtCapacity[tech, capacityType, loc, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyConstructionTimeRule(model, tech,capacityType, loc, time):
    """ construction time of technology, i.e., time that passes between investment and availability"""
    # get parameter object
    params = Parameter.getParameterObject()
    startTimeStep,_     = Technology.getStartEndTimeOfPeriod(tech,time,periodType= "constructionTime",clipToFirstTimeStep=False)
    if startTimeStep in model.setTimeStepsInvest[tech]:
        return (model.builtCapacity[tech,capacityType,loc,time] == model.investedCapacity[tech,capacityType,loc,startTimeStep])
    elif startTimeStep in model.setTimeStepsInvestEntireHorizon[tech]:
        return (model.builtCapacity[tech,capacityType,loc,time] == params.existingInvestedCapacity[tech,capacityType,loc,startTimeStep])
    else:
        return (model.builtCapacity[tech,capacityType,loc,time] == 0)

def constraintTechnologyLifetimeRule(model, tech,capacityType, loc, time):
    """limited lifetime of the technologies"""
    # determine existing capacities
    existingCapacities = Technology.getAvailableExistingQuantity(tech,capacityType,loc,time,typeExistingQuantity="capacity")
    return (model.capacity[tech,capacityType, loc, time]
            == existingCapacities
            + sum(model.builtCapacity[tech,capacityType, loc, previousTime] for previousTime in Technology.getLifetimeRange(tech,time)))

def constraintTechnologyDiffusionLimitRule(model,tech,capacityType ,loc,time):
    """limited technology diffusion based on the existing capacity in the previous year """
    # get parameter object
    params = Parameter.getParameterObject()
    intervalBetweenYears        = EnergySystem.getSystem()["intervalBetweenYears"]
    unboundedMarketShare        = EnergySystem.getSystem()["unboundedMarketShare"]
    knowledgeDepreciationRate   = EnergySystem.getSystem()["knowledgeDepreciationRate"]
    knowledgeSpilloverRate      = EnergySystem.getSystem()["knowledgeSpilloverRate"]
    referenceCarrier            = model.setReferenceCarriers[tech].at(1)
    if params.maxDiffusionRate[tech,time] != np.inf:
        if tech in model.setTransportTechnologies:
            setLocations    = model.setEdges
            setTechnology   = model.setTransportTechnologies
        else:
            setLocations = model.setNodes
            if tech in model.setConversionTechnologies:
                setTechnology = model.setConversionTechnologies
            else:
                setTechnology = model.setStorageTechnologies
        # add built capacity of entire previous horizon
        if params.constructionTimeTechnology[tech] > 0:
            # if technology has lead time, restrict to current capacity
            endTime   = time
        else:
            # else, to capacity in previous time step
            endTime   = time - 1

        rangeTime = range(model.setTimeStepsInvest[tech].at(1),endTime+1)
        # actual years between first invest time step and endTime
        deltaTime       = intervalBetweenYears*(endTime-model.setTimeStepsInvest[tech].at(1))
        # sum up all existing capacities that ever existed and convert to knowledge stock
        totalCapacityKnowledge = (
            sum(
                (params.existingCapacity[tech,capacityType,loc,existingTime]
                 # add spillover from other regions
                + sum(
                    params.existingCapacity[tech, capacityType, otherLoc, existingTime] * knowledgeSpilloverRate
                    for otherLoc in setLocations if otherLoc != loc
                )) *
                (1 - knowledgeDepreciationRate)**(deltaTime + params.lifetimeTechnology[tech] - params.lifetimeExistingTechnology[tech,loc,existingTime])
                for existingTime in model.setExistingTechnologies[tech]
            )
            +
            sum(
                (model.builtCapacity[tech, capacityType, loc, horizonTime]
                 # add spillover from other regions
                + sum(
                    # add spillover from other regions
                    model.builtCapacity[tech, capacityType, loc, horizonTime] * knowledgeSpilloverRate
                    for otherLoc in setLocations if otherLoc != loc
                )) *
                (1 - knowledgeDepreciationRate)**(intervalBetweenYears * (endTime - horizonTime))
                for horizonTime in rangeTime
            )
        )

        totalCapacityAllTechs = sum(
            (Technology.getAvailableExistingQuantity(otherTech, capacityType, loc, time,typeExistingQuantity="capacity")
            + sum(model.builtCapacity[otherTech, capacityType, loc, previousTime] for previousTime in Technology.getLifetimeRange(tech, endTime)))
            for otherTech in setTechnology if model.setReferenceCarriers[otherTech].at(1) == referenceCarrier
        )

        return (
            model.investedCapacity[tech, capacityType, loc, time] <=
            ((1 + params.maxDiffusionRate[tech, time]) ** intervalBetweenYears - 1) * totalCapacityKnowledge
            # add initial market share until which the diffusion rate is unbounded
            + unboundedMarketShare * totalCapacityAllTechs
        )
    else:
        return pe.Constraint.Skip

def constraintCapexYearlyRule(model, tech, capacityType, loc, year):
    """ aggregates the capex of built capacity and of existing capacity """
    system          = EnergySystem.getSystem()
    discountRate    = EnergySystem.getAnalysis()["discountRate"]
    return (model.capexYearly[tech, capacityType, loc, year] == (1 + discountRate) ** (system["intervalBetweenYears"] * (year - model.setTimeStepsYearly.at(1))) *
            (sum(
                model.capex[tech, capacityType, loc, time] *
                (1/(1 + discountRate)) ** (system["intervalBetweenYears"] * (time - model.setTimeStepsYearly.at(1)))
                for time in Technology.getLifetimeRange(tech, year, timeStepType="invest")))
            + Technology.getAvailableExistingQuantity(tech, capacityType, loc, year, typeExistingQuantity="capex",timeStepType="invest"))

def constraintCapexTotalRule(model,year):
    """ sums over all technologies to calculate total capex """
    return(model.capexTotal[year] ==
        sum(
            model.capexYearly[tech, capacityType, loc, year]
            for tech,capacityType,loc in Element.createCustomSet(["setTechnologies","setCapacityTypes","setLocation"]))
    )

def constraintOpexTechnologyRule(model,tech,loc,time):
    """ calculate opex of each technology"""
    # get parameter object
    params = Parameter.getParameterObject()
    referenceCarrier = model.setReferenceCarriers[tech].at(1)
    if tech in model.setConversionTechnologies:
        if referenceCarrier in model.setInputCarriers[tech]:
            referenceFlow = model.inputFlow[tech,referenceCarrier,loc,time]
        else:
            referenceFlow = model.outputFlow[tech,referenceCarrier,loc,time]
    elif tech in model.setTransportTechnologies:
        referenceFlow = model.carrierFlow[tech, loc, time]
    else:
        referenceFlow = model.carrierFlowCharge[tech,loc,time] + model.carrierFlowDischarge[tech,loc,time]
    return(model.opex[tech,loc,time] == params.opexSpecific[tech,loc,time]*referenceFlow)

def constraintCarbonEmissionsTechnologyRule(model,tech,loc,time):
    """ calculate carbon emissions of each technology"""
    # get parameter object
    params = Parameter.getParameterObject()
    referenceCarrier = model.setReferenceCarriers[tech].at(1)
    if tech in model.setConversionTechnologies:
        if referenceCarrier in model.setInputCarriers[tech]:
            referenceFlow = model.inputFlow[tech,referenceCarrier,loc,time]
        else:
            referenceFlow = model.outputFlow[tech,referenceCarrier,loc,time]
    elif tech in model.setTransportTechnologies:
        referenceFlow = model.carrierFlow[tech, loc, time]
    else:
        referenceFlow = model.carrierFlowCharge[tech,loc,time] + model.carrierFlowDischarge[tech,loc,time]
    return(model.carbonEmissionsTechnology[tech,loc,time] == params.carbonIntensityTechnology[tech,loc]*referenceFlow)

def constraintCarbonEmissionsTechnologyTotalRule(model, year):
    """ calculate total carbon emissions of each technology"""
    # get parameter object
    params = Parameter.getParameterObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None,year,"yearly")
    return(
        model.carbonEmissionsTechnologyTotal[year] ==
        sum(
            sum(
                model.carbonEmissionsTechnology[tech,loc,time]*params.timeStepsOperationDuration[tech, time]
                for time in EnergySystem.encodeTimeStep(tech, baseTimeStep, "operation", yearly = True)
            )
            for tech, loc in Element.createCustomSet(["setTechnologies", "setLocation"])
        )
    )

def constraintOpexTotalRule(model,year):
    """ sums over all technologies to calculate total opex """
    # get parameter object
    params = Parameter.getParameterObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None, year, "yearly")
    return(model.opexTotal[year] ==
        sum(
            sum(
                model.opex[tech, loc, time]*params.timeStepsOperationDuration[tech,time]
                for time in EnergySystem.encodeTimeStep(tech, baseTimeStep, "operation", yearly=True)
            )
            for tech,loc in Element.createCustomSet(["setTechnologies","setLocation"])
        )
    )

def constraintMaxLoadRule(model, tech,capacityType, loc, time):
    """Load is limited by the installed capacity and the maximum load factor"""
    # get parameter object
    params = Parameter.getParameterObject()
    referenceCarrier = model.setReferenceCarriers[tech].at(1)
    # get invest time step
    investTimeStep = EnergySystem.convertTimeStepOperation2Invest(tech,time)
    # conversion technology
    if tech in model.setConversionTechnologies:
        if referenceCarrier in model.setInputCarriers[tech]:
            return (model.capacity[tech,capacityType, loc, investTimeStep]*params.maxLoad[tech,capacityType, loc, time] >= model.inputFlow[tech, referenceCarrier, loc, time])
        else:
            return (model.capacity[tech,capacityType, loc, investTimeStep]*params.maxLoad[tech,capacityType, loc, time] >= model.outputFlow[tech, referenceCarrier, loc, time])
    # transport technology
    elif tech in model.setTransportTechnologies:
            return (model.capacity[tech,capacityType, loc, investTimeStep]*params.maxLoad[tech,capacityType, loc, time] >= model.carrierFlow[tech, loc, time])
    # storage technology
    elif tech in model.setStorageTechnologies:
        system = EnergySystem.getSystem()
        # if limit power
        if capacityType == system["setCapacityTypes"][0]:
            return (model.capacity[tech,capacityType, loc, investTimeStep]*params.maxLoad[tech,capacityType, loc, time] >= model.carrierFlowCharge[tech, loc, time] + model.carrierFlowDischarge[tech, loc, time])
        # TODO integrate level storage here as well
        else:
            return pe.Constraint.Skip
        # if limit energy
        # else:
        #     return (model.capacity[tech,capacityType, loc, investTimeStep] * model.maxLoad[tech,capacityType, loc, time] >= model.levelStorage[tech,loc,time])