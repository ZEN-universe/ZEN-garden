"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all storage technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for the storage technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import numpy as np
from .technology import Technology
from ..energy_system import EnergySystem
from ..component import Parameter,Variable,Constraint

class StorageTechnology(Technology):
    # set label
    label           = "setStorageTechnologies"
    location_type    = "setNodes"
    # empty list of elements
    list_of_elements = []
    
    def __init__(self, tech):
        """init storage technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize storage technology {tech}')
        super().__init__(tech)
        # store input data
        self.store_input_data()
        # add StorageTechnology to list
        StorageTechnology.add_element(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get attributes from class <Technology>
        super().store_input_data()
        set_base_time_steps_yearly  = EnergySystem.get_energy_system().set_base_time_steps_yearly
        set_time_steps_yearly      = EnergySystem.get_energy_system().set_time_steps_yearly
        # set attributes for parameters of child class <StorageTechnology>
        self.efficiencyCharge               = self.datainput.extract_input_data("efficiencyCharge",index_sets=["setNodes","set_time_steps"],time_steps= set_time_steps_yearly)
        self.efficiencyDischarge            = self.datainput.extract_input_data("efficiencyDischarge",index_sets=["setNodes","set_time_steps"],time_steps= set_time_steps_yearly)
        self.selfDischarge                  = self.datainput.extract_input_data("selfDischarge",index_sets=["setNodes"])
        # extract existing energy capacity
        self.min_built_capacity_energy         = self.datainput.extract_attribute("minBuiltCapacityEnergy")["value"]
        self.max_built_capacity_energy         = self.datainput.extract_attribute("maxBuiltCapacityEnergy")["value"]
        self.capacity_limit_energy            = self.datainput.extract_input_data("capacityLimitEnergy",index_sets=["setNodes"])
        self.existing_capacity_energy         = self.datainput.extract_input_data("existingCapacityEnergy",index_sets=["setNodes","set_existing_technologies"],column="existingCapacityEnergy")
        self.existing_invested_capacity_energy = self.datainput.extract_input_data("existingInvestedCapacityEnergy", index_sets=["setNodes", "set_time_steps"],time_steps=set_time_steps_yearly)
        self.capex_specific                  = self.datainput.extract_input_data(
            "capexSpecific",index_sets=["setNodes","set_time_steps"],time_steps= set_time_steps_yearly)
        self.capex_specific_energy            = self.datainput.extract_input_data(
            "capexSpecificEnergy",index_sets=["setNodes","set_time_steps"],time_steps=set_time_steps_yearly)
        self.fixed_opex_specific_energy        = self.datainput.extract_input_data("fixedOpexSpecificEnergy", index_sets=["setNodes", "set_time_steps"], time_steps=set_time_steps_yearly)        # annualize capex
        self.convertToAnnualizedCapex()
        # calculate capex of existing capacity
        self.capex_existing_capacity          = self.calculate_capex_of_existing_capacities()
        self.capex_existing_capacity_energy    = self.calculate_capex_of_existing_capacities(storage_energy = True)
        # add min load max load time series for energy
        self.raw_time_series["min_load_energy"] = self.datainput.extract_input_data(
            "minLoadEnergy", index_sets=["setNodes", "set_time_steps"],time_steps=set_base_time_steps_yearly)
        self.raw_time_series["max_load_energy"] = self.datainput.extract_input_data(
            "maxLoadEnergy",index_sets=["setNodes", "set_time_steps"],time_steps=set_base_time_steps_yearly)

    def convertToAnnualizedCapex(self):
        """ this method converts the total capex to annualized capex """
        fractionalAnnuity           = self.calculate_fractional_annuity()
        system                      = EnergySystem.get_system()
        _fraction_year             = system["unaggregatedTimeStepsPerYear"] / system["totalHoursPerYear"]
        # annualize capex
        self.capex_specific          = self.capex_specific        * fractionalAnnuity + self.fixed_opex_specific * _fraction_year
        self.capex_specific_energy    = self.capex_specific_energy  * fractionalAnnuity + self.fixed_opex_specific_energy * _fraction_year

    def calculate_capex_of_single_capacity(self,capacity,index,storage_energy = False):
        """ this method calculates the annualized capex of a single existing capacity. """
        if storage_energy:
            _absoluteCapex = self.capex_specific_energy[index[0]].iloc[0] * capacity
        else:
            _absoluteCapex = self.capex_specific[index[0]].iloc[0] * capacity
        return _absoluteCapex

    def calculate_time_steps_storage_level(self,conducted_tsa):
        """ this method calculates the number of time steps on the storage level, and the sequence in which the storage levels are connected
        conducted_tsa: boolean if the time series were aggregated. If not, the storage level index is the same as the carrier flow indices """
        sequence_time_steps                   = self.sequence_time_steps
        # if time series aggregation was conducted
        if conducted_tsa:
            # calculate connected storage levels, i.e., time steps that are constant for
            IdxLastConnectedStorageLevel        = np.append(np.flatnonzero(np.diff(sequence_time_steps)),len(sequence_time_steps)-1)
            # empty setTimeStep
            self.setTimeStepsStorageLevel       = []
            self.timeStepsStorageLevelDuration  = {}
            time_steps_energy2power               = {}
            self.sequenceTimeStepsStorageLevel  = np.zeros(np.size(sequence_time_steps)).astype(int)
            counterTimeStep                     = 0
            for idxTimeStep,idxStorageLevel in enumerate(IdxLastConnectedStorageLevel):
                self.setTimeStepsStorageLevel.append(idxTimeStep)
                self.timeStepsStorageLevelDuration[idxTimeStep] = len(range(counterTimeStep,idxStorageLevel+1))
                self.sequenceTimeStepsStorageLevel[counterTimeStep:idxStorageLevel+1] = idxTimeStep
                time_steps_energy2power[idxTimeStep]  = sequence_time_steps[idxStorageLevel]
                counterTimeStep                 = idxStorageLevel + 1
        else:
            self.setTimeStepsStorageLevel       = self.setTimeStepsOperation
            self.timeStepsStorageLevelDuration  = self.time_steps_operation_duration
            self.sequenceTimeStepsStorageLevel  = sequence_time_steps
            time_steps_energy2power               = {idx: idx for idx in self.setTimeStepsOperation}

        # add sequence to energy system
        EnergySystem.set_sequence_time_steps(self.name+"StorageLevel",self.sequenceTimeStepsStorageLevel)
        # set the dict time_steps_energy2power
        EnergySystem.set_time_steps_energy2power(self.name, time_steps_energy2power)
        # set the first and last time step of each year
        EnergySystem.set_time_steps_storage_startend(self.name)

    def overwrite_time_steps(self,base_time_steps):
        """ overwrites setTimeStepsStorageLevel """
        super().overwrite_time_steps(base_time_steps)
        setTimeStepsStorageLevel = EnergySystem.encode_time_step(self.name+"StorageLevel", base_time_steps=base_time_steps,time_step_type="operation", yearly=True)
        setattr(self, "setTimeStepsStorageLevel", setTimeStepsStorageLevel.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to StorageTechnology --- ###
    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <StorageTechnology> """
        model = EnergySystem.get_pyomo_model()
        # time steps of storage levels
        model.setTimeStepsStorageLevel = pe.Set(
            model.setStorageTechnologies,
            initialize = cls.get_attribute_of_all_elements("setTimeStepsStorageLevel"),
            doc="Set of time steps of storage levels for all storage technologies. Dimensions: setStorageTechnologies"
        )

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <StorageTechnology> """
        model = EnergySystem.get_pyomo_model()
        
        # time step duration of storage level
        Parameter.add_parameter(
            name ="timeStepsStorageLevelDuration",
            data = EnergySystem.initialize_component(cls,"timeStepsStorageLevelDuration",index_names=["setStorageTechnologies","setTimeStepsStorageLevel"]),
            doc  ="Parameter which specifies the time step duration in StorageLevel for all technologies"
        )
        # efficiency charge
        Parameter.add_parameter(
            name="efficiencyCharge",
            data= EnergySystem.initialize_component(cls,"efficiencyCharge",index_names=["setStorageTechnologies","setNodes","set_time_steps_yearly"]),
            doc = 'efficiency during charging for storage technologies'
        )
        # efficiency discharge
        Parameter.add_parameter(
            name="efficiencyDischarge",
            data= EnergySystem.initialize_component(cls,"efficiencyDischarge",index_names=["setStorageTechnologies","setNodes","set_time_steps_yearly"]),
            doc = 'efficiency during discharging for storage technologies'
        )
        # self discharge
        Parameter.add_parameter(
            name="selfDischarge",
            data= EnergySystem.initialize_component(cls,"selfDischarge",index_names=["setStorageTechnologies","setNodes"]),
            doc = 'self discharge of storage technologies'
        )
        # capex specific
        Parameter.add_parameter(
            name="capexSpecificStorage",
            data= EnergySystem.initialize_component(cls,"capex_specific",index_names=["setStorageTechnologies","set_capacity_types","setNodes","set_time_steps_yearly"],capacity_types=True),
            doc = 'specific capex of storage technologies'
        )

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <StorageTechnology> """

        def carrierFlowBounds(model,tech ,node,time):
            """ return bounds of carrier_flow for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param node: node index
            :param time: time index
            :return bounds: bounds of carrier_flow"""
            # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
            time_step_year = EnergySystem.convert_time_step_operation2invest(tech,time)
            bounds = model.capacity[tech,"power",node,time_step_year].bounds
            return bounds

        model = EnergySystem.get_pyomo_model()
        # flow of carrier on node into storage
        Variable.add_variable(
            model,
            name="carrier_flow_charge",
            index_sets= cls.create_custom_set(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'carrier flow into storage technology on node i and time t'
        )
        # flow of carrier on node out of storage
        Variable.add_variable(
            model,
            name="carrier_flow_discharge",
            index_sets= cls.create_custom_set(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'carrier flow out of storage technology on node i and time t'
        )
        # loss of carrier on node
        Variable.add_variable(
            model,
            name="levelCharge",
            index_sets= cls.create_custom_set(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            domain = pe.NonNegativeReals,
            doc = 'storage level of storage technology ón node in each storage time step'
        )
        
    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <StorageTechnology> """
        model = EnergySystem.get_pyomo_model()
        # Limit storage level
        Constraint.add_constraint(
            model,
            name="constraintStorageLevelMax",
            index_sets= cls.create_custom_set(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            rule = constraintStorageLevelMaxRule,
            doc = 'limit maximum storage level to capacity'
        ) 
        # couple storage levels
        Constraint.add_constraint(
            model,
            name="constraintCoupleStorageLevel",
            index_sets= cls.create_custom_set(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            rule = constraintCoupleStorageLevelRule,
            doc = 'couple subsequent storage levels (time coupling constraints)'
        )
        # Linear Capex
        Constraint.add_constraint(
            model,
            name="constraintStorageTechnologyLinearCapex",
            index_sets= cls.create_custom_set(["setStorageTechnologies","set_capacity_types","setNodes","set_time_steps_yearly"]),
            rule = constraintCapexStorageTechnologyRule,
            doc = 'Capital expenditures for installing storage technology'
        ) 

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls,disjunct, tech,capacity_type, node, time):
        """definition of disjunct constraints if technology is on"""
        model = disjunct.model()
        params = Parameter.get_component_object()
        # get invest time step
        base_time_step = EnergySystem.decode_time_step(tech,time,"operation")
        time_step_year = EnergySystem.encode_time_step(tech,base_time_step,"yearly")
        # disjunct constraints min load charge
        disjunct.constraintMinLoadCharge = pe.Constraint(
            expr=model.carrier_flow_charge[tech, node, time] >= params.min_load[tech,capacity_type,node,time] * model.capacity[tech,capacity_type,node, time_step_year]
        )
        # disjunct constraints min load discharge
        disjunct.constraintMinLoadDischarge = pe.Constraint(
            expr=model.carrier_flow_discharge[tech, node, time] >= params.min_load[tech,capacity_type,node,time] * model.capacity[tech,capacity_type,node, time_step_year]
        )

    @classmethod
    def disjunct_off_technology_rule(cls,disjunct, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        # off charging
        disjunct.constraintNoLoadCharge = pe.Constraint(
            expr=model.carrier_flow_charge[tech, node, time] == 0
        )
        # off discharging
        disjunct.constraintNoLoadDischarge = pe.Constraint(
            expr=model.carrier_flow_discharge[tech, node, time] == 0
        )

    @classmethod
    def getStorageLevelTimeStep(cls,tech,time):
        """ gets current and previous time step of storage level """
        sequenceStorageLevel    = cls.get_attribute_of_specific_element(tech,"sequenceStorageLevel")
        setTimeStepsOperation   = cls.get_attribute_of_specific_element(tech,"setTimeStepsOperation")
        indexCurrentTimeStep    = setTimeStepsOperation.index(time)
        currentLevelTimeStep    = sequenceStorageLevel[indexCurrentTimeStep]
        # if first time step
        if indexCurrentTimeStep == 0:
            previousLevelTimeStep = sequenceStorageLevel[-1]
        # if any other time step
        else:
            previousLevelTimeStep = sequenceStorageLevel[indexCurrentTimeStep-1]
        return currentLevelTimeStep,previousLevelTimeStep

### --- functions with constraint rules --- ###
def constraintStorageLevelMaxRule(model, tech, node, time):
    """limit maximum storage level to capacity"""
    # get invest time step
    element_time_step = EnergySystem.convert_time_step_energy2power(tech,time)
    time_step_year  = EnergySystem.convert_time_step_operation2invest(tech,element_time_step)
    return(model.levelCharge[tech, node, time] <= model.capacity[tech,"energy", node, time_step_year])

def constraintCoupleStorageLevelRule(model, tech, node, time):
    """couple subsequent storage levels (time coupling constraints)"""
    # get parameter object
    params = Parameter.get_component_object()
    element_time_step             = EnergySystem.convert_time_step_energy2power(tech,time)
    # get invest time step
    time_step_year              = EnergySystem.convert_time_step_operation2invest(tech,element_time_step)
    # get corresponding start time step at beginning of the year, if time is last time step in year
    timeStepEnd                 = EnergySystem.get_time_steps_storage_startend(tech,time)
    if timeStepEnd is not None:
        previousLevelTimeStep   = timeStepEnd
    else:
        previousLevelTimeStep   = time-1

    return(
        model.levelCharge[tech, node, time] ==
        model.levelCharge[tech, node, previousLevelTimeStep]*(1-params.selfDischarge[tech,node])**params.timeStepsStorageLevelDuration[tech,time] +
        (model.carrier_flow_charge[tech, node, element_time_step]*params.efficiencyCharge[tech,node,time_step_year] -
        model.carrier_flow_discharge[tech, node, element_time_step]/params.efficiencyDischarge[tech,node,time_step_year])*sum((1-params.selfDischarge[tech,node])**interimTimeStep for interimTimeStep in range(0,params.timeStepsStorageLevelDuration[tech,time]))
    )

def constraintCapexStorageTechnologyRule(model, tech,capacity_type, node, time):
    """ definition of the capital expenditures for the storage technology"""
    # get parameter object
    params = Parameter.get_component_object()
    return (model.capex[tech,capacity_type,node, time] ==
            model.built_capacity[tech,capacity_type,node, time] *
            params.capexSpecificStorage[tech,capacity_type,node, time])
