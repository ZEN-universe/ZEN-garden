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
    list_of_elements = []

    def __init__(self, technology):
        """init generic technology object
        :param object: object of the abstract optimization model
        :param technology: technology that is added to the model"""

        super().__init__(technology)
        # add Technology to list
        Technology.add_element(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # set attributes of technology
        _setLocation    = type(self).getClassLocationType()

        set_base_time_steps_yearly          = EnergySystem.get_energy_system().set_base_time_steps_yearly
        set_time_steps_yearly              = EnergySystem.get_energy_system().set_time_steps_yearly
        self.referenceCarrier           = [self.datainput.extract_attribute("referenceCarrier",skip_warning=True)]
        EnergySystem.set_technology_of_carrier(self.name, self.referenceCarrier)
        self.minBuiltCapacity           = self.datainput.extract_attribute("minBuiltCapacity")["value"]
        self.maxBuiltCapacity           = self.datainput.extract_attribute("maxBuiltCapacity")["value"]
        self.lifetime                   = self.datainput.extract_attribute("lifetime")["value"]
        self.constructionTime           = self.datainput.extract_attribute("constructionTime")["value"]
        # maximum diffusion rate
        self.maxDiffusionRate           = self.datainput.extract_input_data("maxDiffusionRate", index_sets=["set_time_steps"],time_steps=set_time_steps_yearly)

        # add all raw time series to dict
        self.raw_time_series = {}
        self.raw_time_series["minLoad"]       = self.datainput.extract_input_data("minLoad",index_sets=[_setLocation, "set_time_steps"],time_steps=set_base_time_steps_yearly)
        self.raw_time_series["maxLoad"]       = self.datainput.extract_input_data("maxLoad",index_sets=[_setLocation, "set_time_steps"],time_steps=set_base_time_steps_yearly)
        self.raw_time_series["opexSpecific"]  = self.datainput.extract_input_data("opexSpecific",index_sets=[_setLocation,"set_time_steps"],time_steps=set_base_time_steps_yearly)
        # non-time series input data
        self.fixedOpexSpecific          = self.datainput.extract_input_data("fixedOpexSpecific",index_sets=[_setLocation,"set_time_steps"],time_steps=set_time_steps_yearly)
        self.capacityLimit              = self.datainput.extract_input_data("capacityLimit",index_sets=[_setLocation])
        self.carbonIntensityTechnology  = self.datainput.extract_input_data("carbonIntensity",index_sets=[_setLocation])
        # extract existing capacity
        self.setExistingTechnologies    = self.datainput.extract_set_existing_technologies()
        self.existingCapacity           = self.datainput.extract_input_data("existingCapacity",index_sets=[_setLocation,"setExistingTechnologies"])
        self.existingInvestedCapacity   = self.datainput.extract_input_data("existingInvestedCapacity", index_sets=[_setLocation, "set_time_steps"], time_steps=set_time_steps_yearly)
        self.lifetimeExistingTechnology = self.datainput.extract_lifetime_existing_technology("existingCapacity",index_sets=[_setLocation,"setExistingTechnologies"])

    def calculateCapexOfExistingCapacities(self,storage_energy = False):
        """ this method calculates the annualized capex of the existing capacities """
        if storage_energy:
            existingCapacities  = self.existingCapacityEnergy
        else:
            existingCapacities  = self.existingCapacity
        if self.__class__.__name__ == "StorageTechnology":
            existingCapex   = existingCapacities.to_frame().apply(
                lambda _existingCapacity: self.calculateCapexOfSingleCapacity(_existingCapacity.squeeze(),_existingCapacity.name,storage_energy),axis=1)
        else:
            existingCapex   = existingCapacities.to_frame().apply(
                lambda _existingCapacity: self.calculateCapexOfSingleCapacity(_existingCapacity.squeeze(),_existingCapacity.name), axis=1)
        return existingCapex

    def calculateCapexOfSingleCapacity(self,*args):
        """ this method calculates the annualized capex of the existing capacities. Is implemented in child class """
        raise NotImplementedError

    def calculateFractionalAnnuity(self):
        """calculate fraction of annuity to depreciate investment"""
        system              = EnergySystem.get_system()
        _lifetime           = self.lifetime
        _annuity            = 1/_lifetime
        # only account for fraction of year
        _fraction_year     = system["unaggregatedTimeStepsPerYear"] / system["totalHoursPerYear"]
        _fractionalAnnuity  = _annuity * _fraction_year
        return _fractionalAnnuity

    def overwrite_time_steps(self,base_time_steps):
        """ overwrites setTimeStepsOperation """
        setTimeStepsOperation   = EnergySystem.encode_time_step(self.name, base_time_steps=base_time_steps,time_step_type="operation",yearly=True)

        # copy invest time steps
        self.setTimeStepsOperation              = setTimeStepsOperation.squeeze().tolist()

    def add_newly_built_capacity_tech(self,built_capacity,capex,base_time_steps):
        """ adds the newly built capacity to the existing capacity
        :param built_capacity: pd.Series of newly built capacity of technology
        :param capex: pd.Series of capex of newly built capacity of technology
        :param base_time_steps: base time steps of current horizon step """
        system = EnergySystem.get_system()
        # reduce lifetime of existing capacities and add new remaining lifetime
        self.lifetimeExistingTechnology             = (self.lifetimeExistingTechnology - system["intervalBetweenYears"]).clip(lower=0)
        # new capacity
        _timeStepYears                            = EnergySystem.encode_time_step(self.name, base_time_steps, "yearly", yearly=True)
        _newlyBuiltCapacity                         = built_capacity[_timeStepYears].sum(axis=1)
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
                    _energy_string = ""
                # if energy
                else:
                    _energy_string = "Energy"
                _existingCapacity       = getattr(self,"existingCapacity"+_energy_string)
                _capexExistingCapacity  = getattr(self, "capexExistingCapacity" + _energy_string)
                # add new existing capacity
                _existingCapacity                           = _existingCapacity.unstack()
                _existingCapacity[indexNewTechnology]       = _newlyBuiltCapacity.loc[typeCapacity]
                setattr(self,"existingCapacity"+_energy_string,_existingCapacity.stack())
                # calculate capex of existing capacity
                _capexExistingCapacity                      = _capexExistingCapacity.unstack()
                _capexExistingCapacity[indexNewTechnology]  = _capex.loc[typeCapacity]
                setattr(self, "capexExistingCapacity" + _energy_string,_capexExistingCapacity.stack())

    def add_newly_invested_capacity_tech(self,invested_capacity,step_horizon):
        """ adds the newly invested capacity to the list of invested capacity
        :param invested_capacity: pd.Series of newly built capacity of technology
        :param step_horizon: optimization time step """
        system = EnergySystem.get_system()
        _newlyInvestedCapacity = invested_capacity[step_horizon]
        _newlyInvestedCapacity = _newlyInvestedCapacity.fillna(0)
        if not (_newlyInvestedCapacity == 0).all():
            for typeCapacity in list(set(_newlyInvestedCapacity.index.get_level_values(0))):
                # if power
                if typeCapacity == system["setCapacityTypes"][0]:
                    _energy_string = ""
                # if energy
                else:
                    _energy_string = "Energy"
                _existingInvestedCapacity = getattr(self, "existingInvestedCapacity" + _energy_string)
                # add new existing invested capacity
                _existingInvestedCapacity = _existingInvestedCapacity.unstack()
                _existingInvestedCapacity[step_horizon] = _newlyInvestedCapacity.loc[typeCapacity]
                setattr(self, "existingInvestedCapacity" + _energy_string, _existingInvestedCapacity.stack())

    ### --- getter/setter classmethods
    @classmethod
    def getClassLocationType(cls):
        """ returns locationType of class """
        return cls.locationType

    ### --- classmethods
    @classmethod
    def getLifetimeRange(cls, tech, time, time_step_type: str = None):
        """ returns lifetime range of technology. If time_step_type, then converts the yearly time step 'time' to time_step_type """
        if time_step_type:
            base_time_steps   = EnergySystem.decode_time_step(None, time, "yearly")
            timeStepYear  = EnergySystem.encode_time_step(tech, base_time_steps, time_step_type, yearly=True)
        else:
            timeStepYear  = time
        tStart, tEnd = cls.getStartEndTimeOfPeriod(tech, timeStepYear)

        return range(tStart, tEnd + 1)

    @classmethod
    def getAvailableExistingQuantity(cls, tech,capacity_type,loc, time,typeExistingQuantity, time_step_type: str = None):
        """ returns existing quantity of 'tech', that is still available at invest time step 'time'.
        Either capacity or capex.
        :param tech: name of technology
        :param loc: location (node or edge) of existing capacity
        :param time: current time
        :param idExistingCapacity: id of existing capacity
        :return existingQuantity: existing capacity or capex of existing capacity
        """
        params = Parameter.get_component_object()
        system = EnergySystem.get_system()
        discount_rate = EnergySystem.get_analysis()["discount_rate"]
        if time_step_type:
            base_time_steps   = EnergySystem.decode_time_step(None, time, "yearly")
            timeStepYear  = EnergySystem.encode_time_step(tech, base_time_steps, time_step_type, yearly=True)
        else:
            timeStepYear  = time

        model               = EnergySystem.get_pyomo_model()
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
                discountFactor = (1 + discount_rate)**(time*system["intervalBetweenYears"] - yearConstruction)
            else:
                discountFactor = 1
            # if still available at first base time step, add to list
            if tStart == model.set_base_time_steps.at(1) or tStart == timeStepYear:
                existingQuantity += existingVariable[tech,capacity_type, loc, idExistingCapacity]*discountFactor
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
        params  = Parameter.get_component_object()
        model   = EnergySystem.get_pyomo_model()
        system  = EnergySystem.get_system()
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
                if deltaLifetime <= (timeStepYear - model.set_time_steps_yearly.at(1))*system["intervalBetweenYears"]:
                    return timeStepYear
                else:
                    return -1
            periodYearly = params.lifetimeExistingTechnology[tech, loc, idExistingCapacity]
        basePeriod = periodYearly / system["intervalBetweenYears"] * system["unaggregatedTimeStepsPerYear"]
        basePeriod = round(basePeriod, EnergySystem.get_solver()["roundingDecimalPoints"])
        if int(basePeriod) != basePeriod:
            logging.warning(
                f"The period {periodType} of {tech} does not translate to an integer time interval in the base time domain ({basePeriod})")
        # decode to base time steps
        base_time_steps = EnergySystem.decode_time_step(tech, timeStepYear, time_step_type="yearly")
        if len(base_time_steps) == 0:
            return model.set_base_time_steps.at(1), model.set_base_time_steps.at(1) - 1
        base_time_step = base_time_steps[0]

        # if startBaseTimeStep is further in the past than first base time step, use first base time step
        if clipToFirstTimeStep:
            startBaseTimeStep   = int(max(model.set_base_time_steps.at(1), base_time_step - basePeriod + 1))
        else:
            startBaseTimeStep   = int(base_time_step - basePeriod + 1)
        startBaseTimeStep = min(startBaseTimeStep,model.set_base_time_steps.at(-1))
        # if period of existing capacity, then only return the start base time step
        if idExistingCapacity is not None:
            return startBaseTimeStep
        starttimeStepYear     = EnergySystem.encode_time_step(tech, startBaseTimeStep, time_step_type="yearly", yearly=True)[0]

        return starttimeStepYear, endtimeStepYear

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Technology --- ###
    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <Technology> """
        # construct the pe.Sets of the class <Technology>
        model = EnergySystem.get_pyomo_model()

        # conversion technologies
        model.setConversionTechnologies = pe.Set(
            initialize=EnergySystem.get_attribute("setConversionTechnologies"),
            doc='Set of conversion technologies. Subset: setTechnologies')
        # transport technologies
        model.setTransportTechnologies = pe.Set(
            initialize=EnergySystem.get_attribute("setTransportTechnologies"),
            doc='Set of transport technologies. Subset: setTechnologies')
        # storage technologies
        model.setStorageTechnologies = pe.Set(
            initialize=EnergySystem.get_attribute("setStorageTechnologies"),
            doc='Set of storage technologies. Subset: setTechnologies')
        # existing installed technologies
        model.setExistingTechnologies = pe.Set(
            model.setTechnologies,
            initialize=cls.get_attribute_of_all_elements("setExistingTechnologies"),
            doc='Set of existing technologies. Subset: setTechnologies')
        # reference carriers
        model.setReferenceCarriers = pe.Set(
            model.setTechnologies,
            initialize = cls.get_attribute_of_all_elements("referenceCarrier"),
            doc = "set of all reference carriers correspondent to a technology. Dimensions: setTechnologies"
        )
        # add pe.Sets of the child classes
        for subclass in cls.get_all_subclasses():
            subclass.construct_sets()

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <Technology> """
        # construct pe.Param of the class <Technology>

        # existing capacity
        Parameter.add_parameter(
            name="existingCapacity",
            data=EnergySystem.initialize_component(cls,"existingCapacity",index_names=["setTechnologies","setCapacityTypes", "set_location", "setExistingTechnologies"],capacity_types=True),
            doc='Parameter which specifies the existing technology size')
        # existing capacity
        Parameter.add_parameter(
            name="existingInvestedCapacity",
            data=EnergySystem.initialize_component(cls, "existingInvestedCapacity",index_names=["setTechnologies", "setCapacityTypes","set_location", "set_time_steps_yearly_entire_horizon"],capacity_types=True),
            doc='Parameter which specifies the size of the previously invested capacities')
        # minimum capacity
        Parameter.add_parameter(
            name="minBuiltCapacity",
            data= EnergySystem.initialize_component(cls,"minBuiltCapacity",index_names=["setTechnologies", "setCapacityTypes"],capacity_types=True),
            doc = 'Parameter which specifies the minimum technology size that can be installed')
        # maximum capacity
        Parameter.add_parameter(
            name="maxBuiltCapacity",
            data= EnergySystem.initialize_component(cls,"maxBuiltCapacity",index_names=["setTechnologies", "setCapacityTypes"],capacity_types=True),
            doc = 'Parameter which specifies the maximum technology size that can be installed')
        # lifetime existing technologies
        Parameter.add_parameter(
            name="lifetimeExistingTechnology",
            data=EnergySystem.initialize_component(cls,"lifetimeExistingTechnology",index_names=["setTechnologies", "set_location", "setExistingTechnologies"]),
            doc='Parameter which specifies the remaining lifetime of an existing technology')
        # lifetime existing technologies
        Parameter.add_parameter(
            name="capexExistingCapacity",
            data=EnergySystem.initialize_component(cls,"capexExistingCapacity",index_names=["setTechnologies","setCapacityTypes", "set_location", "setExistingTechnologies"],capacity_types=True),
            doc='Parameter which specifies the annualized capex of an existing technology which still has to be paid')
        # lifetime newly built technologies
        Parameter.add_parameter(
            name="lifetimeTechnology",
            data= EnergySystem.initialize_component(cls,"lifetime",index_names=["setTechnologies"]),
            doc = 'Parameter which specifies the lifetime of a newly built technology')
        # constructionTime newly built technologies
        Parameter.add_parameter(
            name="constructionTimeTechnology",
            data=EnergySystem.initialize_component(cls, "constructionTime",index_names=["setTechnologies"]),
            doc='Parameter which specifies the construction time of a newly built technology')
        # maximum diffusion rate, i.e., increase in capacity
        Parameter.add_parameter(
            name="maxDiffusionRate",
            data=EnergySystem.initialize_component(cls, "maxDiffusionRate",index_names=["setTechnologies", "set_time_steps_yearly"]),
            doc="Parameter which specifies the maximum diffusion rate which is the maximum increase in capacity between investment steps")
        # capacityLimit of technologies
        Parameter.add_parameter(
            name="capacityLimitTechnology",
            data= EnergySystem.initialize_component(cls,"capacityLimit",index_names=["setTechnologies","setCapacityTypes","set_location"],capacity_types=True),
            doc = 'Parameter which specifies the capacity limit of technologies')
        # minimum load relative to capacity
        Parameter.add_parameter(
            name="minLoad",
            data= EnergySystem.initialize_component(cls,"minLoad",index_names=["setTechnologies","setCapacityTypes","set_location","setTimeStepsOperation"],capacity_types=True),
            doc = 'Parameter which specifies the minimum load of technology relative to installed capacity')
        # maximum load relative to capacity
        Parameter.add_parameter(
            name="maxLoad",
            data= EnergySystem.initialize_component(cls,"maxLoad",index_names=["setTechnologies","setCapacityTypes","set_location","setTimeStepsOperation"],capacity_types=True),
            doc = 'Parameter which specifies the maximum load of technology relative to installed capacity')
        # specific opex
        Parameter.add_parameter(
            name="opexSpecific",
            data= EnergySystem.initialize_component(cls,"opexSpecific",index_names=["setTechnologies","set_location","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the specific opex')
        # carbon intensity
        Parameter.add_parameter(
            name="carbonIntensityTechnology",
            data= EnergySystem.initialize_component(cls,"carbonIntensityTechnology",index_names=["setTechnologies","set_location"]),
            doc = 'Parameter which specifies the carbon intensity of each technology')
        # add pe.Param of the child classes
        for subclass in cls.get_all_subclasses():
            subclass.construct_params()

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <Technology> """
        def capacityBounds(model,tech,capacity_type, loc, time):
            """ return bounds of capacity for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param capacity_type: either power or energy
            :param loc: location of capacity
            :param time: investment time step
            :return bounds: bounds of capacity"""
            # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
            if tech in techs_on_off:
                system = EnergySystem.get_system()
                params = Parameter.get_component_object()
                if capacity_type == system["setCapacityTypes"][0]:
                    _energy_string = ""
                else:
                    _energy_string = "Energy"
                _existingCapacity           = getattr(params,"existingCapacity"+_energy_string)
                _maxBuiltCapacity           = getattr(params,"maxBuiltCapacity"+_energy_string)
                _capacityLimitTechnology    = getattr(params,"capacityLimitTechnology"+_energy_string)
                existingCapacities = 0
                for idExistingTechnology in model.setExistingTechnologies[tech]:
                    if params.lifetimeExistingTechnology[tech, loc, idExistingTechnology] > params.lifetimeTechnology[tech]:
                        if time > params.lifetimeExistingTechnology[tech, loc, idExistingTechnology] - params.lifetimeTechnology[tech]:
                            existingCapacities += _existingCapacity[tech,capacity_type, loc, idExistingTechnology]
                    elif time <= params.lifetimeExistingTechnology[tech, loc, idExistingTechnology] + 1:
                        existingCapacities  += _existingCapacity[tech,capacity_type, loc, idExistingTechnology]

                maxBuiltCapacity            = len(model.set_time_steps_yearly)*_maxBuiltCapacity[tech,capacity_type]
                maxCapacityLimitTechnology  = _capacityLimitTechnology[tech,capacity_type, loc]
                boundCapacity = min(maxBuiltCapacity + existingCapacities,maxCapacityLimitTechnology + existingCapacities)
                bounds = (0,boundCapacity)
                return(bounds)
            else:
                return(None,None)

        model       = EnergySystem.get_pyomo_model()
        # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
        techs_on_off  = Technology.create_custom_set(["setTechnologies","set_on_off"])[0]
        # construct pe.Vars of the class <Technology>
        # install technology
        Variable.add_variable(
            model,
            name="installTechnology",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            domain = pe.Binary,
            doc = 'installment of a technology at location l and time t')
        # capacity technology
        Variable.add_variable(
            model,
            name="capacity",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            domain = pe.NonNegativeReals,
            bounds = capacityBounds,
            doc = 'size of installed technology at location l and time t')
        # built_capacity technology
        Variable.add_variable(
            model,
            name="built_capacity",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            domain = pe.NonNegativeReals,
            doc = 'size of built technology (invested capacity after construction) at location l and time t')
        # invested_capacity technology
        Variable.add_variable(
            model,
            name="invested_capacity",
            index_sets=cls.create_custom_set(["setTechnologies", "setCapacityTypes", "set_location", "set_time_steps_yearly"]),
            domain=pe.NonNegativeReals,
            doc='size of invested technology at location l and time t')
        # capex of building capacity
        Variable.add_variable(
            model,
            name="capex",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            domain = pe.NonNegativeReals,
            doc = 'capex for building technology at location l and time t')
        # annual capex of having capacity
        Variable.add_variable(
            model,
            name="capexYearly",
            index_sets=cls.create_custom_set(["setTechnologies", "setCapacityTypes", "set_location", "set_time_steps_yearly"]),
            domain=pe.NonNegativeReals,
            doc='annual capex for having technology at location l')
        # total capex
        Variable.add_variable(
            model,
            name="capex_total",
            index_sets=model.set_time_steps_yearly,
            domain = pe.NonNegativeReals,
            doc = 'total capex for installing all technologies in all locations at all times')
        # opex
        Variable.add_variable(
            model,
            name="opex",
            index_sets=cls.create_custom_set(["setTechnologies","set_location","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = "opex for operating technology at location l and time t"
        )
        # total opex
        Variable.add_variable(
            model,
            name="opex_total",
            index_sets=model.set_time_steps_yearly,
            domain = pe.NonNegativeReals,
            doc = "total opex for operating technology at location l and time t"
        )
        # carbon emissions
        Variable.add_variable(
            model,
            name="carbonEmissionsTechnology",
            index_sets=cls.create_custom_set(["setTechnologies","set_location","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = "carbon emissions for operating technology at location l and time t"
        )
        # total carbon emissions technology
        Variable.add_variable(
            model,
            name="carbon_emissions_technology_total",
            index_sets=model.set_time_steps_yearly,
            domain=pe.Reals,
            doc="total carbon emissions for operating technology at location l and time t"
        )

        # add pe.Vars of the child classes
        for subclass in cls.get_all_subclasses():
            subclass.construct_vars()

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <Technology> """
        model = EnergySystem.get_pyomo_model()
        # construct pe.Constraints of the class <Technology>
        #  technology capacityLimit
        Constraint.add_constraint(
            model,
            name="constraintTechnologyCapacityLimit",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            rule = constraintTechnologyCapacityLimitRule,
            doc = 'limited capacity of  technology depending on loc and time'
        )
        # minimum capacity
        Constraint.add_constraint(
            model,
            name="constraintTechnologyMinCapacity",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            rule = constraintTechnologyMinCapacityRule,
            doc = 'min capacity of technology that can be installed'
        )
        # maximum capacity
        Constraint.add_constraint(
            model,
            name="constraintTechnologyMaxCapacity",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            rule = constraintTechnologyMaxCapacityRule,
            doc = 'max capacity of technology that can be installed'
        )
        # construction period
        Constraint.add_constraint(
            model,
            name="constraintTechnologyConstructionTime",
            index_sets=cls.create_custom_set(["setTechnologies", "setCapacityTypes", "set_location", "set_time_steps_yearly"]),
            rule=constraintTechnologyConstructionTimeRule,
            doc='lead time in which invested technology is constructed'
        )
        # lifetime
        Constraint.add_constraint(
            model,
            name="constraintTechnologyLifetime",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","set_time_steps_yearly"]),
            rule = constraintTechnologyLifetimeRule,
            doc = 'max capacity of  technology that can be installed'
        )
        # limit diffusion rate
        Constraint.add_constraint(
            model,
            name="constraintTechnologyDiffusionLimit",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location", "set_time_steps_yearly"]),
            rule=constraintTechnologyDiffusionLimitRule,
            doc="Limits the newly built capacity by the existing knowledge stock")
        # limit max load by installed capacity
        Constraint.add_constraint(
            model,
            name="constraintMaxLoad",
            index_sets=cls.create_custom_set(["setTechnologies","setCapacityTypes","set_location","setTimeStepsOperation"]),
            rule = constraintMaxLoadRule,
            doc = 'limit max load by installed capacity'
        )
        # annual capex of having capacity
        Constraint.add_constraint(
            model,
            name="constraintCapexYearly",
            index_sets=cls.create_custom_set(["setTechnologies", "setCapacityTypes", "set_location", "set_time_steps_yearly"]),
            rule=constraintCapexYearlyRule,
            doc='annual capex of having capacity of technology.'
        )
        # total capex of all technologies
        Constraint.add_constraint(
            model,
            name="constraintCapexTotal",
            index_sets= model.set_time_steps_yearly,
            rule = constraintCapexTotalRule,
            doc = 'total capex of all technology that can be installed.'
        )
        # calculate opex
        Constraint.add_constraint(
            model,
            name="constraintOpexTechnology",
            index_sets=cls.create_custom_set(["setTechnologies","set_location","setTimeStepsOperation"]),
            rule = constraintOpexTechnologyRule,
            doc = "opex for each technology at each location and time step"
        )
        # total opex of all technologies
        Constraint.add_constraint(
            model,
            name="constraintOpexTotal",
            index_sets= model.set_time_steps_yearly,
            rule = constraintOpexTotalRule,
            doc = 'total opex of all technology that are operated.'
        )
        # carbon emissions of technologies
        Constraint.add_constraint(
            model,
            name="constraintCarbonEmissionsTechnology",
            index_sets=cls.create_custom_set(["setTechnologies","set_location","setTimeStepsOperation"]),
            rule = constraintCarbonEmissionsTechnologyRule,
            doc = "carbon emissions for each technology at each location and time step"
        )
        # total carbon emissions of technologies
        Constraint.add_constraint(
            model,
            name="constraintCarbonEmissionsTechnologyTotal",
            index_sets= model.set_time_steps_yearly,
            rule=constraintCarbonEmissionsTechnologyTotalRule,
            doc="total carbon emissions for each technology at each location and time step"
        )

        # disjunct if technology is on
        Constraint.add_constraint(
            model,
            name="disjunctOnTechnology",
            index_sets= cls.create_custom_set(["setTechnologies","set_on_off", "setCapacityTypes","set_location","setTimeStepsOperation"]),
            rule = cls.disjunctOnTechnologyRule,
            doc = "disjunct to indicate that technology is on",
            constraintType = "Disjunct"
        )
        # disjunct if technology is off
        Constraint.add_constraint(
            model,
            name="disjunctOffTechnology",
            index_sets= cls.create_custom_set(["setTechnologies","set_on_off", "setCapacityTypes","set_location","setTimeStepsOperation"]),
            rule = cls.disjunctOffTechnologyRule,
            doc = "disjunct to indicate that technology is off",
            constraintType = "Disjunct"
        )
        # disjunction
        Constraint.add_constraint(
            model,
            name="disjunctionDecisionOnOffTechnology",
            index_sets= cls.create_custom_set(["setTechnologies","set_on_off", "setCapacityTypes","set_location","setTimeStepsOperation"]),
            rule = cls.expressionLinkDisjunctsRule,
            doc = "disjunction to link the on off disjuncts",
            constraintType = "Disjunction"
        )

        # add pe.Constraints of the child classes
        for subclass in cls.get_all_subclasses():
            subclass.construct_constraints()

    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech, capacity_type, loc, time):
        """definition of disjunct constraints if technology is On
        iterate through all subclasses to find corresponding implementation of disjunct constraints """
        for subclass in cls.get_all_subclasses():
            if tech in subclass.get_all_names_of_elements():
                # disjunct is defined in corresponding subclass
                subclass.disjunctOnTechnologyRule(disjunct,tech, capacity_type,loc,time)
                break

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, capacity_type, loc, time):
        """definition of disjunct constraints if technology is off
        iterate through all subclasses to find corresponding implementation of disjunct constraints """
        for subclass in cls.get_all_subclasses():
            if tech in subclass.get_all_names_of_elements():
                # disjunct is defined in corresponding subclass
                subclass.disjunctOffTechnologyRule(disjunct,tech, capacity_type,loc,time)
                break

    @classmethod
    def expressionLinkDisjunctsRule(cls,model, tech, capacity_type, loc, time):
        """ link disjuncts for technology is on and technology is off """
        return ([model.disjunctOnTechnology[tech, capacity_type,loc,time],model.disjunctOffTechnology[tech, capacity_type,loc,time]])

### --- constraint rules --- ###
#%% Constraint rules pre-defined in Technology class
def constraintTechnologyCapacityLimitRule(model, tech,capacity_type, loc, time):
    """limited capacityLimit of technology"""
    # get parameter object
    params = Parameter.get_component_object()
    if params.capacityLimitTechnology[tech,capacity_type, loc] != np.inf:
        existingCapacities = Technology.getAvailableExistingQuantity(tech, capacity_type, loc, time,typeExistingQuantity="capacity")
        if existingCapacities < params.capacityLimitTechnology[tech, capacity_type, loc]:
            return (params.capacityLimitTechnology[tech,capacity_type, loc] >= model.capacity[tech,capacity_type, loc, time])
        else:
            return (model.built_capacity[tech, capacity_type, loc, time] == 0)
    else:
        return pe.Constraint.Skip

def constraintTechnologyMinCapacityRule(model, tech,capacity_type, loc, time):
    """ min capacity expansion of technology."""
    # get parameter object
    params = Parameter.get_component_object()
    if params.minBuiltCapacity[tech,capacity_type] != 0:
        return (params.minBuiltCapacity[tech,capacity_type] * model.installTechnology[tech,capacity_type, loc, time] <= model.built_capacity[tech,capacity_type, loc, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyMaxCapacityRule(model, tech,capacity_type, loc, time):
    """max capacity expansion of technology"""
    # get parameter object
    params = Parameter.get_component_object()
    system = EnergySystem.get_system()
    if params.maxBuiltCapacity[tech,capacity_type] != np.inf:
        return (params.maxBuiltCapacity[tech,capacityType] * model.installTechnology[tech,capacityType, loc, time] >= model.built_capacity[tech,capacityType, loc, time])
    elif system['DoubleCapexTransport'] and tech in system["setTransportTechnologies"] and model.maxCapacity[tech,capacityType] != np.inf:
        return (params.maxCapacity[tech, capacityType] * model.installTechnology[tech, capacityType, loc, time] >= model.built_capacity[tech, capacityType, loc, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyConstructionTimeRule(model, tech,capacityType, loc, time):
    """ construction time of technology, i.e., time that passes between investment and availability"""
    # get parameter object
    params = Parameter.get_component_object()
    startTimeStep,_     = Technology.getStartEndTimeOfPeriod(tech,time,periodType= "constructionTime",clipToFirstTimeStep=False)
    if startTimeStep in model.set_time_steps_yearly:
        return (model.built_capacity[tech,capacityType,loc,time] == model.invested_capacity[tech,capacityType,loc,startTimeStep])
    elif startTimeStep in model.set_time_steps_yearly_entire_horizon:
        return (model.built_capacity[tech,capacityType,loc,time] == params.existingInvestedCapacity[tech,capacityType,loc,startTimeStep])
    else:
        return (model.built_capacity[tech,capacityType,loc,time] == 0)

def constraintTechnologyLifetimeRule(model, tech,capacityType, loc, time):
    """limited lifetime of the technologies"""
    # determine existing capacities
    existingCapacities = Technology.getAvailableExistingQuantity(tech,capacityType,loc,time,typeExistingQuantity="capacity")
    return (model.capacity[tech,capacityType, loc, time]
            == existingCapacities
            + sum(model.built_capacity[tech,capacityType, loc, previousTime] for previousTime in Technology.getLifetimeRange(tech,time)))

def constraintTechnologyDiffusionLimitRule(model,tech,capacityType ,loc,time):
    """limited technology diffusion based on the existing capacity in the previous year """
    # get parameter object
    params = Parameter.get_component_object()
    interval_between_years        = EnergySystem.get_system()["intervalBetweenYears"]
    unboundedMarketShare        = EnergySystem.get_system()["unboundedMarketShare"]
    knowledgeDepreciationRate   = EnergySystem.get_system()["knowledgeDepreciationRate"]
    knowledgeSpilloverRate      = EnergySystem.get_system()["knowledgeSpilloverRate"]
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

        rangeTime = range(model.set_time_steps_yearly.at(1),endTime+1)
        # actual years between first invest time step and endTime
        deltaTime       = interval_between_years*(endTime-model.set_time_steps_yearly.at(1))
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
                (model.built_capacity[tech, capacityType, loc, horizonTime]
                 # add spillover from other regions
                + sum(
                    # add spillover from other regions
                    model.built_capacity[tech, capacityType, loc, horizonTime] * knowledgeSpilloverRate
                    for otherLoc in setLocations if otherLoc != loc
                )) *
                (1 - knowledgeDepreciationRate)**(interval_between_years * (endTime - horizonTime))
                for horizonTime in rangeTime
            )
        )

        totalCapacityAllTechs = sum(
            (Technology.getAvailableExistingQuantity(otherTech, capacityType, loc, time,typeExistingQuantity="capacity")
            + sum(model.built_capacity[otherTech, capacityType, loc, previousTime] for previousTime in Technology.getLifetimeRange(tech, endTime)))
            for otherTech in setTechnology if model.setReferenceCarriers[otherTech].at(1) == referenceCarrier
        )

        return (
            model.invested_capacity[tech, capacityType, loc, time] <=
            ((1 + params.maxDiffusionRate[tech, time]) ** interval_between_years - 1) * totalCapacityKnowledge
            # add initial market share until which the diffusion rate is unbounded
            + unboundedMarketShare * totalCapacityAllTechs
        )
    else:
        return pe.Constraint.Skip

def constraintCapexYearlyRule(model, tech, capacityType, loc, year):
    """ aggregates the capex of built capacity and of existing capacity """
    system          = EnergySystem.get_system()
    discount_rate    = EnergySystem.get_analysis()["discount_rate"]
    return (model.capexYearly[tech, capacityType, loc, year] == (1 + discount_rate) ** (system["intervalBetweenYears"] * (year - model.set_time_steps_yearly.at(1))) *
            (sum(
                model.capex[tech, capacityType, loc, time] *
                (1/(1 + discount_rate)) ** (system["intervalBetweenYears"] * (time - model.set_time_steps_yearly.at(1)))
                for time in Technology.getLifetimeRange(tech, year, time_step_type="yearly")))
            + Technology.getAvailableExistingQuantity(tech, capacityType, loc, year, typeExistingQuantity="capex",time_step_type="yearly"))

def constraintCapexTotalRule(model,year):
    """ sums over all technologies to calculate total capex """
    return(model.capex_total[year] ==
        sum(
            model.capexYearly[tech, capacityType, loc, year]
            for tech,capacityType,loc in Element.create_custom_set(["setTechnologies","setCapacityTypes","set_location"])[0])
    )

def constraintOpexTechnologyRule(model,tech,loc,time):
    """ calculate opex of each technology"""
    # get parameter object
    params = Parameter.get_component_object()
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
    params = Parameter.get_component_object()
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
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None,year,"yearly")
    return(
        model.carbon_emissions_technology_total[year] ==
        sum(
            sum(
                model.carbonEmissionsTechnology[tech,loc,time]*params.timeStepsOperationDuration[tech, time]
                for time in EnergySystem.encode_time_step(tech, base_time_step, "operation", yearly = True)
            )
            for tech, loc in Element.create_custom_set(["setTechnologies", "set_location"])[0]
        )
    )

def constraintOpexTotalRule(model,year):
    """ sums over all technologies to calculate total opex """
    # get parameter object
    params = Parameter.get_component_object()
    base_time_step = EnergySystem.decode_time_step(None, year, "yearly")
    return(model.opex_total[year] ==
        sum(
            sum(
                model.opex[tech, loc, time]*params.timeStepsOperationDuration[tech,time]
                for time in EnergySystem.encode_time_step(tech, base_time_step, "operation", yearly=True)
            )
            for tech,loc in Element.create_custom_set(["setTechnologies","set_location"])[0]
        )
    )

def constraintMaxLoadRule(model, tech,capacityType, loc, time):
    """Load is limited by the installed capacity and the maximum load factor"""
    # get parameter object
    params = Parameter.get_component_object()
    referenceCarrier = model.setReferenceCarriers[tech].at(1)
    # get invest time step
    timeStepYear = EnergySystem.convert_time_step_operation2invest(tech,time)
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
        system = EnergySystem.get_system()
        # if limit power
        if capacityType == system["setCapacityTypes"][0]:
            return (model.capacity[tech,capacityType, loc, timeStepYear]*params.maxLoad[tech,capacityType, loc, time] >= model.carrierFlowCharge[tech, loc, time] + model.carrierFlowDischarge[tech, loc, time])
        # TODO integrate level storage here as well
        else:
            return pe.Constraint.Skip
        # if limit energy
        # else:
        #     return (model.capacity[tech,capacityType, loc, timeStepYear] * model.maxLoad[tech,capacityType, loc, time] >= model.levelStorage[tech,loc,time])
