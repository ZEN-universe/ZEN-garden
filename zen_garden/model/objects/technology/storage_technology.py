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


class StorageTechnology(Technology):
    # set label
    label = "set_storage_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, energy_system):
        """init storage technology object
        :param tech: name of added technology
        :param energy_system: The energy system the element is part of"""

        logging.info(f'Initialize storage technology {tech}')
        super().__init__(tech, energy_system)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        set_base_time_steps_yearly = self.energy_system.set_base_time_steps_yearly
        set_time_steps_yearly = self.energy_system.set_time_steps_yearly
        # set attributes for parameters of child class <StorageTechnology>
        self.efficiency_charge = self.data_input.extract_input_data("efficiency_charge", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.efficiency_discharge = self.data_input.extract_input_data("efficiency_discharge", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.self_discharge = self.data_input.extract_input_data("self_discharge", index_sets=["set_nodes"])
        self.initial_storage_level = self.data_input.extract_input_data("initial_storage_level",index_sets=["set_nodes"])
        # extract existing energy capacity
        self.min_built_capacity_energy = self.data_input.extract_attribute("min_built_capacity_energy")["value"]
        self.max_built_capacity_energy = self.data_input.extract_attribute("max_built_capacity_energy")["value"]
        self.capacity_limit_energy = self.data_input.extract_input_data("capacity_limit_energy", index_sets=["set_nodes"])
        self.existing_capacity_energy = self.data_input.extract_input_data("existing_capacity_energy", index_sets=["set_nodes", "set_existing_technologies"])
        self.existing_invested_capacity_energy = self.data_input.extract_input_data("existing_invested_capacity_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.capex_specific_energy = self.data_input.extract_input_data("capex_specific_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.fixed_opex_specific_energy = self.data_input.extract_input_data("fixed_opex_specific_energy", index_sets=["set_nodes", "set_time_steps_yearly"],
                                                                            time_steps=set_time_steps_yearly)  # annualize capex
        self.convert_to_annualized_capex()
        # calculate capex of existing capacity
        self.capex_existing_capacity = self.calculate_capex_of_existing_capacities()
        self.capex_existing_capacity_energy = self.calculate_capex_of_existing_capacities(storage_energy=True)
        # add min load max load time series for energy
        self.raw_time_series["min_load_energy"] = self.data_input.extract_input_data("min_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["max_load_energy"] = self.data_input.extract_input_data("max_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)

    def convert_to_annualized_capex(self):
        """ this method converts the total capex to annualized capex """
        fractional_annuity = self.calculate_fractional_annuity()
        system = self.energy_system.system
        _fraction_year = system["unaggregated_time_steps_per_year"] / system["total_hours_per_year"]
        # annualize capex
        self.capex_specific = self.capex_specific * fractional_annuity + self.fixed_opex_specific * _fraction_year
        self.capex_specific_energy = self.capex_specific_energy * fractional_annuity + self.fixed_opex_specific_energy * _fraction_year

    def calculate_capex_of_single_capacity(self, capacity, index, storage_energy=False):
        """ this method calculates the annualized capex of a single existing capacity. """
        if storage_energy:
            _absolute_capex = self.capex_specific_energy[index[0]].iloc[0] * capacity
        else:
            _absolute_capex = self.capex_specific[index[0]].iloc[0] * capacity
        return _absolute_capex

    def calculate_time_steps_storage_level(self, conducted_tsa):
        """ this method calculates the number of time steps on the storage level, and the sequence in which the storage levels are connected
        conducted_tsa: boolean if the time series were aggregated. If not, the storage level index is the same as the carrier flow indices """
        sequence_time_steps = self.sequence_time_steps
        # if time series aggregation was conducted
        if conducted_tsa:
            # calculate connected storage levels, i.e., time steps that are constant for
            idx_last_connected_storage_level = np.append(np.flatnonzero(np.diff(sequence_time_steps)), len(sequence_time_steps) - 1)
            # empty setTimeStep
            self.set_time_steps_storage_level = []
            self.time_steps_storage_level_duration = {}
            time_steps_energy2power = {}
            self.sequence_time_steps_storage_level = np.zeros(np.size(sequence_time_steps)).astype(int)
            counter_time_step = 0
            for idx_time_step, idx_storage_level in enumerate(idx_last_connected_storage_level):
                self.set_time_steps_storage_level.append(idx_time_step)
                self.time_steps_storage_level_duration[idx_time_step] = len(range(counter_time_step, idx_storage_level + 1))
                self.sequence_time_steps_storage_level[counter_time_step:idx_storage_level + 1] = idx_time_step
                time_steps_energy2power[idx_time_step] = sequence_time_steps[idx_storage_level]
                counter_time_step = idx_storage_level + 1
        else:
            self.set_time_steps_storage_level = self.set_time_steps_operation
            self.time_steps_storage_level_duration = self.time_steps_operation_duration
            self.sequence_time_steps_storage_level = sequence_time_steps
            time_steps_energy2power = {idx: idx for idx in self.set_time_steps_operation}

        # add sequence to energy system
        self.energy_system.set_sequence_time_steps(self.name + "_storage_level", self.sequence_time_steps_storage_level)
        # set the dict time_steps_energy2power
        self.energy_system.set_time_steps_energy2power(self.name, time_steps_energy2power)
        # set the first and last time step of each year
        self.energy_system.set_time_steps_storage_startend(self.name)

    def overwrite_time_steps(self, base_time_steps):
        """ overwrites set_time_steps_storage_level """
        super().overwrite_time_steps(base_time_steps)
        set_time_steps_storage_level = self.energy_system.encode_time_step(self.name + "_storage_level", base_time_steps=base_time_steps, time_step_type="operation", yearly=True)
        setattr(self, "set_time_steps_storage_level", set_time_steps_storage_level.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to StorageTechnology --- ###
    @classmethod
    def construct_sets(cls, energy_system: EnergySystem):
        """ constructs the pe.Sets of the class <StorageTechnology>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model
        # time steps of storage levels
        model.set_time_steps_storage_level = pe.Set(model.set_storage_technologies, initialize=energy_system.get_attribute_of_all_elements(cls, "set_time_steps_storage_level"),
            doc="Set of time steps of storage levels for all storage technologies. Dimensions: set_storage_technologies")

    @classmethod
    def construct_params(cls, energy_system: EnergySystem):
        """ constructs the pe.Params of the class <StorageTechnology>
        :param energy_system: The Energy system to add everything"""

        # time step duration of storage level
        energy_system.parameters.add_parameter(name="time_steps_storage_level_duration",
            data=energy_system.initialize_component(cls, "time_steps_storage_level_duration", index_names=["set_storage_technologies", "set_time_steps_storage_level"]),
            doc="Parameter which specifies the time step duration in StorageLevel for all technologies")
        # efficiency charge
        energy_system.parameters.add_parameter(name="efficiency_charge",
            data=energy_system.initialize_component(cls, "efficiency_charge", index_names=["set_storage_technologies", "set_nodes", "set_time_steps_yearly"]),
            doc='efficiency during charging for storage technologies')
        # efficiency discharge
        energy_system.parameters.add_parameter(name="efficiency_discharge",
            data=energy_system.initialize_component(cls, "efficiency_discharge", index_names=["set_storage_technologies", "set_nodes", "set_time_steps_yearly"]),
            doc='efficiency during discharging for storage technologies')
        # initial storage level
        energy_system.parameters.add_parameter(name="initial_storage_level",
            data=energy_system.initialize_component(cls, "initial_storage_level", index_names=["set_storage_technologies", "set_nodes"]),
            doc='initial storage level of storage technologies.')
        # self discharge
        energy_system.parameters.add_parameter(name="self_discharge",
            data=energy_system.initialize_component(cls, "self_discharge", index_names=["set_storage_technologies", "set_nodes"]),
            doc='self discharge of storage technologies')
        # capex specific
        energy_system.parameters.add_parameter(name="capex_specific_storage",
            data=energy_system.initialize_component(cls, "capex_specific", index_names=["set_storage_technologies", "set_capacity_types", "set_nodes", "set_time_steps_yearly"], capacity_types=True),
            doc='specific capex of storage technologies')

    @classmethod
    def construct_vars(cls, energy_system: EnergySystem):
        """ constructs the pe.Vars of the class <StorageTechnology>
        :param energy_system: The Energy system to add everything"""

        def carrier_flow_bounds(model, tech, node, time):
            """ return bounds of carrier_flow for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param node: node index
            :param time: time index
            :return bounds: bounds of carrier_flow"""
            # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
            time_step_year = energy_system.convert_time_step_operation2invest(tech, time)
            bounds = model.capacity[tech, "power", node, time_step_year].bounds
            return bounds

        model = energy_system.pyomo_model
        # flow of carrier on node into storage
        energy_system.variables.add_variable(model, name="carrier_flow_charge", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            bounds=carrier_flow_bounds, doc='carrier flow into storage technology on node i and time t')
        # flow of carrier on node out of storage
        energy_system.variables.add_variable(model, name="carrier_flow_discharge", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            bounds=carrier_flow_bounds, doc='carrier flow out of storage technology on node i and time t')
        # loss of carrier on node
        energy_system.variables.add_variable(model, name="level_charge", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage_level"], energy_system), domain=pe.NonNegativeReals,
            doc='storage level of storage technology ón node in each storage time step')

    @classmethod
    def construct_constraints(cls, energy_system: EnergySystem):
        """ constructs the pe.Constraints of the class <StorageTechnology>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model
        rules = StorageTechnologyRules(energy_system)
        # Limit storage level
        energy_system.constraints.add_constraint(model, name="constraint_storage_level_max", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage_level"], energy_system),
            rule=rules.constraint_storage_level_max_rule, doc='limit maximum storage level to capacity')
        # couple storage levels
        energy_system.constraints.add_constraint(model, name="constraint_couple_storage_level", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage_level"], energy_system),
            rule=rules.constraint_couple_storage_level_rule, doc='couple subsequent storage levels (time coupling constraints)')
        # initial storage level
        energy_system.constraints.add_constraint(model, name="constraint_initial_storage_level", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage_level"], energy_system),
            rule=rules.constraint_initial_storage_level_rule, doc='sets initial storage level')
        # Linear Capex
        energy_system.constraints.add_constraint(model, name="constraint_storage_technology_capex",
            index_sets=cls.create_custom_set(["set_storage_technologies", "set_capacity_types", "set_nodes", "set_time_steps_yearly"], energy_system), rule=rules.constraint_storage_technology_capex_rule,
            doc='Capital expenditures for installing storage technology')

        # defines disjuncts if technology on/off

    @classmethod
    def disjunct_on_technology_rule(cls, energy_system, disjunct, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is on"""
        model = disjunct.model()
        params = energy_system.parameters
        # get invest time step
        base_time_step = energy_system.decode_time_step(tech, time, "operation")
        time_step_year = energy_system.encode_time_step(tech, base_time_step, "yearly")
        # disjunct constraints min load charge
        disjunct.constraint_min_load_charge = pe.Constraint(
            expr=model.carrier_flow_charge[tech, node, time] >= params.min_load[tech, capacity_type, node, time] * model.capacity[tech, capacity_type, node, time_step_year])
        # disjunct constraints min load discharge
        disjunct.constraint_min_load_discharge = pe.Constraint(
            expr=model.carrier_flow_discharge[tech, node, time] >= params.min_load[tech, capacity_type, node, time] * model.capacity[tech, capacity_type, node, time_step_year])

    @classmethod
    def disjunct_off_technology_rule(cls, disjunct, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        # off charging
        disjunct.constraint_no_load_charge = pe.Constraint(expr=model.carrier_flow_charge[tech, node, time] == 0)
        # off discharging
        disjunct.constraint_no_load_discharge = pe.Constraint(expr=model.carrier_flow_discharge[tech, node, time] == 0)

    @classmethod
    def getStorageLevelTimeStep(cls, energy_system, tech, time):
        """ gets current and previous time step of storage level """
        sequence_storage_level = energy_system.get_attribute_of_specific_element(cls, tech, "sequence_storage_level")
        set_time_steps_operation = energy_system.get_attribute_of_specific_element(cls, tech, "set_time_steps_operation")
        index_current_time_step = set_time_steps_operation.index(time)
        current_level_time_step = sequence_storage_level[index_current_time_step]
        # if first time step
        if index_current_time_step == 0:
            previous_level_time_step = sequence_storage_level[-1]
        # if any other time step
        else:
            previous_level_time_step = sequence_storage_level[index_current_time_step - 1]
        return current_level_time_step, previous_level_time_step


class StorageTechnologyRules:
    """
    Rules for the StorageTechnology class
    """

    def __init__(self, energy_system: EnergySystem):
        """
        Inits the rules for a given EnergySystem
        :param energy_system: The EnergySystem
        """

        self.energy_system = energy_system

    ### --- functions with constraint rules --- ###
    def constraint_storage_level_max_rule(self, model, tech, node, time):
        """limit maximum storage level to capacity"""
        # get invest time step
        element_time_step = self.energy_system.convert_time_step_energy2power(tech, time)
        time_step_year = self.energy_system.convert_time_step_operation2invest(tech, element_time_step)
        return (model.level_charge[tech, node, time] <= model.capacity[tech, "energy", node, time_step_year])

    def constraint_couple_storage_level_rule(self, model, tech, node, time):
        """couple subsequent storage levels (time coupling constraints)"""
        # get parameter object
        params = self.energy_system.parameters
        system = self.energy_system.system
        element_time_step = self.energy_system.convert_time_step_energy2power(tech, time)
        # get invest time step
        time_step_year = self.energy_system.convert_time_step_operation2invest(tech, element_time_step)
        # get corresponding start time step at beginning of the year, if time is last time step in year
        time_step_end = self.energy_system.get_time_steps_storage_startend(tech, time)
        enforce_periodicity = True
        if time_step_end is not None:
            previous_level_time_step = time_step_end
            if not system["storage_periodicity"]:
                enforce_periodicity = False
        else:
            previous_level_time_step = time - 1
        if enforce_periodicity:
            return (model.level_charge[tech, node, time] == model.level_charge[tech, node, previous_level_time_step] * (1 - params.self_discharge[tech, node]) ** params.time_steps_storage_level_duration[
                tech, time] + (model.carrier_flow_charge[tech, node, element_time_step] * params.efficiency_charge[tech, node, time_step_year] - model.carrier_flow_discharge[tech, node, element_time_step] /
                               params.efficiency_discharge[tech, node, time_step_year]) * sum(
                (1 - params.self_discharge[tech, node]) ** interimTimeStep for interimTimeStep in range(0, params.time_steps_storage_level_duration[tech, time])))
        else:
            return pe.Constraint.Skip

    def constraint_initial_storage_level_rule(self,model, tech, node, time):
        """ sets storage level in first time step of the year to initial storage level.
        Skip if initial storage level is inf"""
        # get parameter object
        params = self.energy_system.parameters
        if params.initial_storage_level[tech, node] != np.inf:
            # get corresponding start time step at beginning of the year, if time is last time step in year
            time_step_end = self.energy_system.get_time_steps_storage_startend(tech, time)
            # if time_step_end exists, time is first time step of the year
            if time_step_end:
                element_time_step = self.energy_system.convert_time_step_energy2power(tech, time)
                invest_time_step = self.energy_system.convert_time_step_operation2invest(tech, element_time_step)
                return (
                        model.level_charge[tech, node, time] ==
                        params.initial_storage_level[tech, node] * model.capacity[tech, "energy", node, invest_time_step]
                )
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip

    def constraint_storage_technology_capex_rule(self, model, tech, capacity_type, node, time):
        """ definition of the capital expenditures for the storage technology"""
        # get parameter object
        params = self.energy_system.parameters
        return (model.capex[tech, capacity_type, node, time] == model.built_capacity[tech, capacity_type, node, time] * params.capex_specific_storage[tech, capacity_type, node, time])
