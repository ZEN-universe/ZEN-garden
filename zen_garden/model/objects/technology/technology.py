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
from ..element import Element
from ..energy_system import EnergySystem
from ..component import Parameter,Variable,Constraint

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
        setTimeStepsYearly              = EnergySystem.getEnergySystem().setTimeStepsYearly
        self.referenceCarrier           = [self.dataInput.extractAttributeData("referenceCarrier",skipWarning=True)]
        EnergySystem.setTechnologyOfCarrier(self.name, self.referenceCarrier)
        self.minBuiltCapacity           = self.dataInput.extractAttributeData("minBuiltCapacity")["value"]
        self.maxBuiltCapacity           = self.dataInput.extractAttributeData("maxBuiltCapacity")["value"]
        self.lifetime                   = self.dataInput.extractAttributeData("lifetime")["value"]
        self.constructionTime           = self.dataInput.extractAttributeData("constructionTime")["value"]
        # maximum diffusion rate
        self.maxDiffusionRate           = self.dataInput.extractInputData("maxDiffusionRate", indexSets=["setTimeSteps"],timeSteps=setTimeStepsYearly)

        # add all raw time series to dict
        self.rawTimeSeries = {}
        self.rawTimeSeries["minLoad"]       = self.dataInput.extractInputData("minLoad",indexSets=[_setLocation, "setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["maxLoad"]       = self.dataInput.extractInputData("maxLoad",indexSets=[_setLocation, "setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["opexSpecific"]  = self.dataInput.extractInputData("opexSpecific",indexSets=[_setLocation,"setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        # non-time series input data
        self.fixedOpexSpecific          = self.dataInput.extractInputData("fixedOpexSpecific",indexSets=[_setLocation,"setTimeSteps"],timeSteps=setTimeStepsYearly)
        self.capacityLimit              = self.dataInput.extractInputData("capacityLimit",indexSets=[_setLocation])
        self.carbonIntensityTechnology  = self.dataInput.extractInputData("carbonIntensity",indexSets=[_setLocation])
        # extract existing capacity
        self.setExistingTechnologies    = self.dataInput.extractSetExistingTechnologies()
        self.existingCapacity           = self.dataInput.extractInputData("existingCapacity",indexSets=[_setLocation,"setExistingTechnologies"])
        self.existingInvestedCapacity   = self.dataInput.extractInputData("existingInvestedCapacity", indexSets=[_setLocation, "setTimeSteps"], timeSteps=setTimeStepsYearly)
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
        """ overwrites setTimeStepsOperation """
        setTimeStepsOperation   = EnergySystem.encodeTimeStep(self.name, baseTimeSteps=baseTimeSteps,timeStepType="operation",yearly=True)

        # copy invest time steps
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
        _timeStepYears                            = EnergySystem.encodeTimeStep(self.name, baseTimeSteps, "yearly", yearly=True)
        _newlyBuiltCapacity                         = builtCapacity[_timeStepYears].sum(axis=1)
        _capex                                      = capex[_timeStepYears].sum(axis=1)
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
            timeStepYear  = EnergySystem.encodeTimeStep(tech, baseTimeSteps, timeStepType, yearly=True)
        else:
            timeStepYear  = time
        tStart, tEnd = cls.getStartEndTimeOfPeriod(tech, timeStepYear)

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
        params = Parameter.getComponentObject()
        system = EnergySystem.getSystem()
        discountRate = EnergySystem.getAnalysis()["discountRate"]
        if timeStepType:
            baseTimeSteps   = EnergySystem.decodeTimeStep(None, time, "yearly")
            timeStepYear  = EnergySystem.encodeTimeStep(tech, baseTimeSteps, timeStepType, yearly=True)
        else:
            timeStepYear  = time

        model               = EnergySystem.getConcreteModel()
        existingQuantity = 0
        if typeExistingQuantity == "capacity":
            existingVariable = params.existingCapacity
        elif typeExistingQuantity == "capex":
            existingVariable = params.capexExistingCapacity
        else:
            raise KeyError(f"Wrong type of existing quantity {typeExistingQuantity}")

        for idExistingCapacity in model.setExistingTechnologies[tech]:
            tStart  = cls.getStartEndTimeOfPeriod(tech, timeStepYear, idExistingCapacity=idExistingCapacity,loc= loc)
            # discount existing capex
            if typeExistingQuantity == "capex":
                yearConstruction = max(0,time*system["intervalBetweenYears"] - params.lifetimeTechnology[tech] + params.lifetimeExistingTechnology[tech,loc,idExistingCapacity])
                discountFactor = (1 + discountRate)**(time*system["intervalBetweenYears"] - yearConstruction)
            else:
                discountFactor = 1
            # if still available at first base time step, add to list
            if tStart == model.setBaseTimeSteps.at(1) or tStart == timeStepYear:
                existingQuantity += existingVariable[tech,capacityType, loc, idExistingCapacity]*discountFactor
        return existingQuantity

    @classmethod
    def getStartEndTimeOfPeriod(cls, tech, timeStepYear,periodType = "lifetime",clipToFirstTimeStep = True, idExistingCapacity = None,loc = None):
        """ counts back the period (either lifetime of constructionTime) back to get the start invest time step and returns starttimeStepYear
        :param tech: name of technology
        :param timeStepYear: current investment time step
        :param periodType: "lifetime" if lifetime is counted backwards, "constructionTime" if construction time is counted backwards
        :param clipToFirstTimeStep: boolean to clip the time step to first time step if time step too far in the past
        :param idExistingCapacity: id of existing capacity
        :param loc: location (node or edge) of existing capacity
        :return beganInPast: boolean if the period began before the first optimization step
        :return starttimeStepYear,endtimeStepYear: start and end of period in invest time step domain"""

        # get model and system
        params  = Parameter.getComponentObject()
        model   = EnergySystem.getConcreteModel()
        system  = EnergySystem.getSystem()
        # get which period to count backwards
        if periodType == "lifetime":
            periodTime = params.lifetimeTechnology
        elif periodType == "constructionTime":
            periodTime = params.constructionTimeTechnology
        else:
            raise NotImplemented(f"getStartEndOfPeriod not yet implemented for {periodType}")
        # get endtimeStepYear
        if not isinstance(timeStepYear, np.ndarray):
            endtimeStepYear = timeStepYear
        elif len(timeStepYear) == 1:
            endtimeStepYear = timeStepYear[0]
        # if more than one investment time step
        else:
            endtimeStepYear = timeStepYear[-1]
            timeStepYear    = timeStepYear[0]
        # convert period to interval of base time steps
        if idExistingCapacity is None:
            periodYearly = periodTime[tech]
        else:
            deltaLifetime = params.lifetimeExistingTechnology[tech, loc, idExistingCapacity] - periodTime[tech]
            if deltaLifetime >= 0:
                if deltaLifetime <= (timeStepYear - model.setTimeStepsYearly.at(1))*system["intervalBetweenYears"]:
                    return timeStepYear
                else:
                    return -1
            periodYearly = params.lifetimeExistingTechnology[tech, loc, idExistingCapacity]
        basePeriod = periodYearly / system["intervalBetweenYears"] * system["unaggregatedTimeStepsPerYear"]
        basePeriod = round(basePeriod, EnergySystem.getSolver()["roundingDecimalPoints"])
        if int(basePeriod) != basePeriod:
            logging.warning(
                f"The period {periodType} of {tech} does not translate to an integer time interval in the base time domain ({basePeriod})")
        # decode to base time steps
        baseTimeSteps = EnergySystem.decodeTimeStep(tech, timeStepYear, timeStepType="yearly")
        if len(baseTimeSteps) == 0:
            return model.setBaseTimeSteps.at(1), model.setBaseTimeSteps.at(1) - 1
        baseTimeStep = baseTimeSteps[0]

        # if startBaseTimeStep is further in the past than first base time step, use first base time step
        if clipToFirstTimeStep:
            startBaseTimeStep   = int(max(model.setBaseTimeSteps.at(1), baseTimeStep - basePeriod + 1))
        else:
            startBaseTimeStep   = int(baseTimeStep - basePeriod + 1)
        startBaseTimeStep = min(startBaseTimeStep,model.setBaseTimeSteps.at(-1))
        # if period of existing capacity, then only return the start base time step
        if idExistingCapacity is not None:
            return startBaseTimeStep
        starttimeStepYear     = EnergySystem.encodeTimeStep(tech, startBaseTimeStep, timeStepType="yearly", yearly=True)[0]

        return starttimeStepYear, endtimeStepYear

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

        # existing capacity
        Parameter.addParameter(
            name="existingCapacity",
            data=EnergySystem.initializeComponent(cls,"existingCapacity",indexNames=["setTechnologies","setCapacityTypes", "setLocation", "setExistingTechnologies"],capacityTypes=True),
            doc='Parameter which specifies the existing technology size')
        # existing capacity
        Parameter.addParameter(
            name="existingInvestedCapacity",
            data=EnergySystem.initializeComponent(cls, "existingInvestedCapacity",indexNames=["setTechnologies", "setCapacityTypes","setLocation", "setTimeStepsYearlyEntireHorizon"],capacityTypes=True),
            doc='Parameter which specifies the size of the previously invested capacities')
        # minimum capacity
        Parameter.addParameter(
            name="minBuiltCapacity",
            data= EnergySystem.initializeComponent(cls,"minBuiltCapacity",indexNames=["setTechnologies", "setCapacityTypes"],capacityTypes=True),
            doc = 'Parameter which specifies the minimum technology size that can be installed')
        # maximum capacity
        Parameter.addParameter(
            name="maxBuiltCapacity",
            data= EnergySystem.initializeComponent(cls,"maxBuiltCapacity",indexNames=["setTechnologies", "setCapacityTypes"],capacityTypes=True),
            doc = 'Parameter which specifies the maximum technology size that can be installed')
        # lifetime existing technologies
        Parameter.addParameter(
            name="lifetimeExistingTechnology",
            data=EnergySystem.initializeComponent(cls,"lifetimeExistingTechnology",indexNames=["setTechnologies", "setLocation", "setExistingTechnologies"]),
            doc='Parameter which specifies the remaining lifetime of an existing technology')
        # lifetime existing technologies
        Parameter.addParameter(
            name="capexExistingCapacity",
            data=EnergySystem.initializeComponent(cls,"capexExistingCapacity",indexNames=["setTechnologies","setCapacityTypes", "setLocation", "setExistingTechnologies"],capacityTypes=True),
            doc='Parameter which specifies the annualized capex of an existing technology which still has to be paid')
        # lifetime newly built technologies
        Parameter.addParameter(
            name="lifetimeTechnology",
            data= EnergySystem.initializeComponent(cls,"lifetime",indexNames=["setTechnologies"]),
            doc = 'Parameter which specifies the lifetime of a newly built technology')
        # constructionTime newly built technologies
        Parameter.addParameter(
            name="constructionTimeTechnology",
            data=EnergySystem.initializeComponent(cls, "constructionTime",indexNames=["setTechnologies"]),
            doc='Parameter which specifies the construction time of a newly built technology')
        # maximum diffusion rate, i.e., increase in capacity
        Parameter.addParameter(
            name="maxDiffusionRate",
            data=EnergySystem.initializeComponent(cls, "maxDiffusionRate",indexNames=["setTechnologies", "setTimeStepsYearly"]),
            doc="Parameter which specifies the maximum diffusion rate which is the maximum increase in capacity between investment steps")
        # capacityLimit of technologies
        Parameter.addParameter(
            name="capacityLimitTechnology",
            data= EnergySystem.initializeComponent(cls,"capacityLimit",indexNames=["setTechnologies","setCapacityTypes","setLocation"],capacityTypes=True),
            doc = 'Parameter which specifies the capacity limit of technologies')
        # minimum load relative to capacity
        Parameter.addParameter(
            name="minLoad",
            data= EnergySystem.initializeComponent(cls,"minLoad",indexNames=["setTechnologies","setCapacityTypes","setLocation","setTimeStepsOperation"],capacityTypes=True),
            doc = 'Parameter which specifies the minimum load of technology relative to installed capacity')
        # maximum load relative to capacity
        Parameter.addParameter(
            name="maxLoad",
            data= EnergySystem.initializeComponent(cls,"maxLoad",indexNames=["setTechnologies","setCapacityTypes","setLocation","setTimeStepsOperation"],capacityTypes=True),
            doc = 'Parameter which specifies the maximum load of technology relative to installed capacity')
        # specific opex
        Parameter.addParameter(
            name="opexSpecific",
            data= EnergySystem.initializeComponent(cls,"opexSpecific",indexNames=["setTechnologies","setLocation","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the specific opex')
        # carbon intensity
        Parameter.addParameter(
            name="carbonIntensityTechnology",
            data= EnergySystem.initializeComponent(cls,"carbonIntensityTechnology",indexNames=["setTechnologies","setLocation"]),
            doc = 'Parameter which specifies the carbon intensity of each technology')
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
            if tech in techsOnOff:
                system = EnergySystem.getSystem()
                params = Parameter.getComponentObject()
                if capacityType == system["setCapacityTypes"][0]:
                    _energyString = ""
                else:
                    _energyString = "Energy"
                _existingCapacity           = getattr(params,"existingCapacity"+_energyString)
                _maxBuiltCapacity           = getattr(params,"maxBuiltCapacity"+_energyString)
                _capacityLimitTechnology    = getattr(params,"capacityLimitTechnology"+_energyString)
                existingCapacities = 0
                for idExistingTechnology in model.setExistingTechnologies[tech]:
                    if params.lifetimeExistingTechnology[tech, loc, idExistingTechnology] > params.lifetimeTechnology[tech]:
                        if time > params.lifetimeExistingTechnology[tech, loc, idExistingTechnology] - params.lifetimeTechnology[tech]:
                            existingCapacities += _existingCapacity[tech,capacityType, loc, idExistingTechnology]
                    elif time <= params.lifetimeExistingTechnology[tech, loc, idExistingTechnology] + 1:
                        existingCapacities  += _existingCapacity[tech,capacityType, loc, idExistingTechnology]

                maxBuiltCapacity            = len(model.setTimeStepsYearly)*_maxBuiltCapacity[tech,capacityType]
                maxCapacityLimitTechnology  = _capacityLimitTechnology[tech,capacityType, loc]
                boundCapacity = min(maxBuiltCapacity + existingCapacities,maxCapacityLimitTechnology + existingCapacities)
                bounds = (0,boundCapacity)
                return(bounds)
            else:
                return(None,None)

        model       = EnergySystem.getConcreteModel()
        # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
        techsOnOff  = Technology.createCustomSet(["setTechnologies","setOnOff"])[0]
        # construct pe.Vars of the class <Technology>
        # install technology
        Variable.addVariable(
            model,
            name="installTechnology",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            domain = pe.Binary,
            doc = 'installment of a technology at location l and time t')
        # capacity technology
        Variable.addVariable(
            model,
            name="capacity",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            domain = pe.NonNegativeReals,
            bounds = capacityBounds,
            doc = 'size of installed technology at location l and time t')
        # builtCapacity technology
        Variable.addVariable(
            model,
            name="builtCapacity",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            domain = pe.NonNegativeReals,
            doc = 'size of built technology (invested capacity after construction) at location l and time t')
        # investedCapacity technology
        Variable.addVariable(
            model,
            name="investedCapacity",
            indexSets=cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsYearly"]),
            domain=pe.NonNegativeReals,
            doc='size of invested technology at location l and time t')
        # capex of building capacity
        Variable.addVariable(
            model,
            name="capex",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            domain = pe.NonNegativeReals,
            doc = 'capex for building technology at location l and time t')
        # annual capex of having capacity
        Variable.addVariable(
            model,
            name="capexYearly",
            indexSets=cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsYearly"]),
            domain=pe.NonNegativeReals,
            doc='annual capex for having technology at location l')
        # total capex
        Variable.addVariable(
            model,
            name="capexTotal",
            indexSets=model.setTimeStepsYearly,
            domain = pe.NonNegativeReals,
            doc = 'total capex for installing all technologies in all locations at all times')
        # opex
        Variable.addVariable(
            model,
            name="opex",
            indexSets=cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = "opex for operating technology at location l and time t"
        )
        # total opex
        Variable.addVariable(
            model,
            name="opexTotal",
            indexSets=model.setTimeStepsYearly,
            domain = pe.NonNegativeReals,
            doc = "total opex for operating technology at location l and time t"
        )
        # carbon emissions
        Variable.addVariable(
            model,
            name="carbonEmissionsTechnology",
            indexSets=cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = "carbon emissions for operating technology at location l and time t"
        )
        # total carbon emissions technology
        Variable.addVariable(
            model,
            name="carbonEmissionsTechnologyTotal",
            indexSets=model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="total carbon emissions for operating technology at location l and time t"
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
        Constraint.addConstraint(
            model,
            name="constraintTechnologyCapacityLimit",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            rule = constraintTechnologyCapacityLimitRule,
            doc = 'limited capacity of  technology depending on loc and time'
        )
        # minimum capacity
        Constraint.addConstraint(
            model,
            name="constraintTechnologyMinCapacity",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            rule = constraintTechnologyMinCapacityRule,
            doc = 'min capacity of technology that can be installed'
        )
        # maximum capacity
        Constraint.addConstraint(
            model,
            name="constraintTechnologyMaxCapacity",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            rule = constraintTechnologyMaxCapacityRule,
            doc = 'max capacity of technology that can be installed'
        )
        # construction period
        Constraint.addConstraint(
            model,
            name="constraintTechnologyConstructionTime",
            indexSets=cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsYearly"]),
            rule=constraintTechnologyConstructionTimeRule,
            doc='lead time in which invested technology is constructed'
        )
        # lifetime
        Constraint.addConstraint(
            model,
            name="constraintTechnologyLifetime",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsYearly"]),
            rule = constraintTechnologyLifetimeRule,
            doc = 'max capacity of  technology that can be installed'
        )
        # limit diffusion rate
        Constraint.addConstraint(
            model,
            name="constraintTechnologyDiffusionLimit",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation", "setTimeStepsYearly"]),
            rule=constraintTechnologyDiffusionLimitRule,
            doc="Limits the newly built capacity by the existing knowledge stock")
        # limit max load by installed capacity
        Constraint.addConstraint(
            model,
            name="constraintMaxLoad",
            indexSets=cls.createCustomSet(["setTechnologies","setCapacityTypes","setLocation","setTimeStepsOperation"]),
            rule = constraintMaxLoadRule,
            doc = 'limit max load by installed capacity'
        )
        # annual capex of having capacity
        Constraint.addConstraint(
            model,
            name="constraintCapexYearly",
            indexSets=cls.createCustomSet(["setTechnologies", "setCapacityTypes", "setLocation", "setTimeStepsYearly"]),
            rule=constraintCapexYearlyRule,
            doc='annual capex of having capacity of technology.'
        )
        # total capex of all technologies
        Constraint.addConstraint(
            model,
            name="constraintCapexTotal",
            indexSets= model.setTimeStepsYearly,
            rule = constraintCapexTotalRule,
            doc = 'total capex of all technology that can be installed.'
        )
        # calculate opex
        Constraint.addConstraint(
            model,
            name="constraintOpexTechnology",
            indexSets=cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            rule = constraintOpexTechnologyRule,
            doc = "opex for each technology at each location and time step"
        )
        # total opex of all technologies
        Constraint.addConstraint(
            model,
            name="constraintOpexTotal",
            indexSets= model.setTimeStepsYearly,
            rule = constraintOpexTotalRule,
            doc = 'total opex of all technology that are operated.'
        )
        # carbon emissions of technologies
        Constraint.addConstraint(
            model,
            name="constraintCarbonEmissionsTechnology",
            indexSets=cls.createCustomSet(["setTechnologies","setLocation","setTimeStepsOperation"]),
            rule = constraintCarbonEmissionsTechnologyRule,
            doc = "carbon emissions for each technology at each location and time step"
        )
        # total carbon emissions of technologies
        Constraint.addConstraint(
            model,
            name="constraintCarbonEmissionsTechnologyTotal",
            indexSets= model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsTechnologyTotalRule,
            doc="total carbon emissions for each technology at each location and time step"
        )

        # disjunct if technology is on
        Constraint.addConstraint(
            model,
            name="disjunctOnTechnology",
            indexSets= cls.createCustomSet(["setTechnologies","setOnOff", "setCapacityTypes","setLocation","setTimeStepsOperation"]),
            rule = cls.disjunctOnTechnologyRule,
            doc = "disjunct to indicate that technology is on",
            constraintType = "Disjunct"
        )
        # disjunct if technology is off
        Constraint.addConstraint(
            model,
            name="disjunctOffTechnology",
            indexSets= cls.createCustomSet(["setTechnologies","setOnOff", "setCapacityTypes","setLocation","setTimeStepsOperation"]),
            rule = cls.disjunctOffTechnologyRule,
            doc = "disjunct to indicate that technology is off",
            constraintType = "Disjunct"
        )
        # disjunction
        Constraint.addConstraint(
            model,
            name="disjunctionDecisionOnOffTechnology",
            indexSets= cls.createCustomSet(["setTechnologies","setOnOff", "setCapacityTypes","setLocation","setTimeStepsOperation"]),
            rule = cls.expressionLinkDisjunctsRule,
            doc = "disjunction to link the on off disjuncts",
            constraintType = "Disjunction"
        )

        # add pe.Constraints of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructConstraints()

    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech, capacityType, loc, time):
        """definition of disjunct constraints if technology is On
        iterate through all subclasses to find corresponding implementation of disjunct constraints """
        for subclass in cls.getAllSubclasses():
            if tech in subclass.getAllNamesOfElements():
                # disjunct is defined in corresponding subclass
                subclass.disjunctOnTechnologyRule(disjunct,tech, capacityType,loc,time)
                break

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, capacityType, loc, time):
        """definition of disjunct constraints if technology is off
        iterate through all subclasses to find corresponding implementation of disjunct constraints """
        for subclass in cls.getAllSubclasses():
            if tech in subclass.getAllNamesOfElements():
                # disjunct is defined in corresponding subclass
                subclass.disjunctOffTechnologyRule(disjunct,tech, capacityType,loc,time)
                break

    @classmethod
    def expressionLinkDisjunctsRule(cls,model, tech, capacityType, loc, time):
        """ link disjuncts for technology is on and technology is off """
        return ([model.disjunctOnTechnology[tech, capacityType,loc,time],model.disjunctOffTechnology[tech, capacityType,loc,time]])

### --- constraint rules --- ###
#%% Constraint rules pre-defined in Technology class
def constraintTechnologyCapacityLimitRule(model, tech,capacityType, loc, time):
    """limited capacityLimit of technology"""
    # get parameter object
    params = Parameter.getComponentObject()
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
    params = Parameter.getComponentObject()
    if params.minBuiltCapacity[tech,capacityType] != 0:
        return (params.minBuiltCapacity[tech,capacityType] * model.installTechnology[tech,capacityType, loc, time] <= model.builtCapacity[tech,capacityType, loc, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyMaxCapacityRule(model, tech,capacityType, loc, time):
    """max capacity expansion of technology"""
    # get parameter object
    params = Parameter.getComponentObject()
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
    params = Parameter.getComponentObject()
    startTimeStep,_     = Technology.getStartEndTimeOfPeriod(tech,time,periodType= "constructionTime",clipToFirstTimeStep=False)
    if startTimeStep in model.setTimeStepsYearly:
        return (model.builtCapacity[tech,capacityType,loc,time] == model.investedCapacity[tech,capacityType,loc,startTimeStep])
    elif startTimeStep in model.setTimeStepsYearlyEntireHorizon:
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
    params = Parameter.getComponentObject()
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

        rangeTime = range(model.setTimeStepsYearly.at(1),endTime+1)
        # actual years between first invest time step and endTime
        deltaTime       = intervalBetweenYears*(endTime-model.setTimeStepsYearly.at(1))
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
                for time in Technology.getLifetimeRange(tech, year, timeStepType="yearly")))
            + Technology.getAvailableExistingQuantity(tech, capacityType, loc, year, typeExistingQuantity="capex",timeStepType="yearly"))

def constraintCapexTotalRule(model,year):
    """ sums over all technologies to calculate total capex """
    return(model.capexTotal[year] ==
        sum(
            model.capexYearly[tech, capacityType, loc, year]
            for tech,capacityType,loc in Element.createCustomSet(["setTechnologies","setCapacityTypes","setLocation"])[0])
    )

def constraintOpexTechnologyRule(model,tech,loc,time):
    """ calculate opex of each technology"""
    # get parameter object
    params = Parameter.getComponentObject()
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
    params = Parameter.getComponentObject()
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
    params = Parameter.getComponentObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None,year,"yearly")
    return(
        model.carbonEmissionsTechnologyTotal[year] ==
        sum(
            sum(
                model.carbonEmissionsTechnology[tech,loc,time]*params.timeStepsOperationDuration[tech, time]
                for time in EnergySystem.encodeTimeStep(tech, baseTimeStep, "operation", yearly = True)
            )
            for tech, loc in Element.createCustomSet(["setTechnologies", "setLocation"])[0]
        )
    )

def constraintOpexTotalRule(model,year):
    """ sums over all technologies to calculate total opex """
    # get parameter object
    params = Parameter.getComponentObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None, year, "yearly")
    return(model.opexTotal[year] ==
        sum(
            sum(
                model.opex[tech, loc, time]*params.timeStepsOperationDuration[tech,time]
                for time in EnergySystem.encodeTimeStep(tech, baseTimeStep, "operation", yearly=True)
            )
            for tech,loc in Element.createCustomSet(["setTechnologies","setLocation"])[0]
        )
    )

def constraintMaxLoadRule(model, tech,capacityType, loc, time):
    """Load is limited by the installed capacity and the maximum load factor"""
    # get parameter object
    params = Parameter.getComponentObject()
    referenceCarrier = model.setReferenceCarriers[tech].at(1)
    # get invest time step
    timeStepYear = EnergySystem.convertTimeStepOperation2Invest(tech,time)
    # conversion technology
    if tech in model.setConversionTechnologies:
        if referenceCarrier in model.setInputCarriers[tech]:
            return (model.capacity[tech,capacityType, loc, timeStepYear]*params.maxLoad[tech,capacityType, loc, time] >= model.inputFlow[tech, referenceCarrier, loc, time])
        else:
            return (model.capacity[tech,capacityType, loc, timeStepYear]*params.maxLoad[tech,capacityType, loc, time] >= model.outputFlow[tech, referenceCarrier, loc, time])
    # transport technology
    elif tech in model.setTransportTechnologies:
            return (model.capacity[tech,capacityType, loc, timeStepYear]*params.maxLoad[tech,capacityType, loc, time] >= model.carrierFlow[tech, loc, time])
    # storage technology
    elif tech in model.setStorageTechnologies:
        system = EnergySystem.getSystem()
        # if limit power
        if capacityType == system["setCapacityTypes"][0]:
            return (model.capacity[tech,capacityType, loc, timeStepYear]*params.maxLoad[tech,capacityType, loc, time] >= model.carrierFlowCharge[tech, loc, time] + model.carrierFlowDischarge[tech, loc, time])
        # TODO integrate level storage here as well
        else:
            return pe.Constraint.Skip
        # if limit energy
        # else:
        #     return (model.capacity[tech,capacityType, loc, timeStepYear] * model.maxLoad[tech,capacityType, loc, time] >= model.levelStorage[tech,loc,time])
