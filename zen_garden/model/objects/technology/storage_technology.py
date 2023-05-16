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

import numpy as np
import xarray as xr

from zen_garden.utils import linexpr_from_tuple_np
from .technology import Technology
from ..component import ZenIndex, IndexSet
from ..element import Element


class StorageTechnology(Technology):
    # set label
    label = "set_storage_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, optimization_setup):
        """init storage technology object
        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of """

        logging.info(f'Initialize storage technology {tech}')
        super().__init__(tech, optimization_setup)
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
        # extract existing energy capacity
        self.capacity_addition_min_energy = self.data_input.extract_attribute("capacity_addition_min_energy")["value"]
        self.capacity_addition_max_energy = self.data_input.extract_attribute("capacity_addition_max_energy")["value"]
        self.capacity_limit_energy = self.data_input.extract_input_data("capacity_limit_energy", index_sets=["set_nodes"])
        self.capacity_existing_energy = self.data_input.extract_input_data("capacity_existing_energy", index_sets=["set_nodes", "set_technologies_existing"])
        self.capacity_investment_existing_energy = self.data_input.extract_input_data("capacity_investment_existing_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.capex_specific_energy = self.data_input.extract_input_data("capex_specific_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        self.opex_specific_fixed_energy = self.data_input.extract_input_data("opex_specific_fixed_energy", index_sets=["set_nodes", "set_time_steps_yearly"],
                                                                            time_steps=set_time_steps_yearly)
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()
        self.capex_capacity_existing_energy = self.calculate_capex_of_capacities_existing(storage_energy=True)
        # add min load max load time series for energy
        self.raw_time_series["min_load_energy"] = self.data_input.extract_input_data("min_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)
        self.raw_time_series["max_load_energy"] = self.data_input.extract_input_data("max_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps=set_base_time_steps_yearly)

    def convert_to_fraction_of_capex(self):
        """ this method converts the total capex to fraction of capex, depending on how many hours per year are calculated """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.opex_specific_fixed_energy = self.opex_specific_fixed_energy * fraction_year
        self.capex_specific = self.capex_specific * fraction_year
        self.capex_specific_energy = self.capex_specific_energy * fraction_year

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
        self.energy_system.time_steps.set_sequence_time_steps(self.name + "_storage_level", self.sequence_time_steps_storage_level)
        # set the dict time_steps_energy2power
        self.energy_system.time_steps.set_time_steps_energy2power(self.name, time_steps_energy2power)
        # set the first and last time step of each year
        self.energy_system.time_steps.set_time_steps_storage_startend(self.name, self.optimization_setup.system)

    def overwrite_time_steps(self, base_time_steps):
        """ overwrites set_time_steps_storage_level """
        super().overwrite_time_steps(base_time_steps)
        set_time_steps_storage_level = self.energy_system.time_steps.encode_time_step(self.name + "_storage_level", base_time_steps=base_time_steps, time_step_type="operation", yearly=True)
        setattr(self, "set_time_steps_storage_level", set_time_steps_storage_level.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to StorageTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <StorageTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        # time steps of storage levels
        optimization_setup.sets.add_set(name="set_time_steps_storage_level", data=optimization_setup.get_attribute_of_all_elements(cls, "set_time_steps_storage_level"),
                                        doc="Set of time steps of storage levels for all storage technologies. Dimensions: set_storage_technologies",
                                        index_set="set_storage_technologies")

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <StorageTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """

        # time step duration of storage level
        optimization_setup.parameters.add_parameter(name="time_steps_storage_level_duration",
            data=optimization_setup.initialize_component(cls, "time_steps_storage_level_duration", index_names=["set_storage_technologies", "set_time_steps_storage_level"]),
            doc="Parameter which specifies the time step duration in StorageLevel for all technologies")
        # efficiency charge
        optimization_setup.parameters.add_parameter(name="efficiency_charge",
            data=optimization_setup.initialize_component(cls, "efficiency_charge", index_names=["set_storage_technologies", "set_nodes", "set_time_steps_yearly"]),
            doc='efficiency during charging for storage technologies')
        # efficiency discharge
        optimization_setup.parameters.add_parameter(name="efficiency_discharge",
            data=optimization_setup.initialize_component(cls, "efficiency_discharge", index_names=["set_storage_technologies", "set_nodes", "set_time_steps_yearly"]),
            doc='efficiency during discharging for storage technologies')
        # self discharge
        optimization_setup.parameters.add_parameter(name="self_discharge",
            data=optimization_setup.initialize_component(cls, "self_discharge", index_names=["set_storage_technologies", "set_nodes"]),
            doc='self discharge of storage technologies')
        # capex specific
        optimization_setup.parameters.add_parameter(name="capex_specific_storage",
            data=optimization_setup.initialize_component(cls, "capex_specific", index_names=["set_storage_technologies", "set_capacity_types", "set_nodes", "set_time_steps_yearly"], capacity_types=True),
            doc='specific capex of storage technologies')

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <StorageTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """

        model = optimization_setup.model
        variables = optimization_setup.variables
        sets = optimization_setup.sets

        def flow_storage_bounds(index_values, index_list):
            """ return bounds of carrier_flow for bigM expression
            :param index_values: list of tuples with the index values
            :param index_list: The names of the indices
            :return bounds: bounds of carrier_flow"""

            # get the arrays
            tech_arr, node_arr, time_arr = sets.tuple_to_arr(index_values, index_list)
            # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
            time_step_year = xr.DataArray([optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech, time) for tech, time in zip(tech_arr.data, time_arr.data)])
            lower = model.variables["capacity"].lower.loc[tech_arr, "power", node_arr, time_step_year].data
            upper = model.variables["capacity"].upper.loc[tech_arr, "power", node_arr, time_step_year].data
            return np.stack([lower, upper], axis=-1)

        # flow of carrier on node into storage
        index_values, index_names = cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_operation"], optimization_setup)
        bounds = flow_storage_bounds(index_values, index_names)
        variables.add_variable(model, name="flow_storage_charge", index_sets=(index_values, index_names),
            bounds=bounds, doc='carrier flow into storage technology on node i and time t')
        # flow of carrier on node out of storage
        variables.add_variable(model, name="flow_storage_discharge", index_sets=(index_values, index_names),
            bounds=bounds, doc='carrier flow out of storage technology on node i and time t')
        # loss of carrier on node
        variables.add_variable(model, name="storage_level", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage_level"], optimization_setup), bounds=(0, np.inf),
            doc='storage level of storage technology ón node in each storage time step')

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <StorageTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        rules = StorageTechnologyRules(optimization_setup)
        # Limit storage level
        constraints.add_constraint_block(model, name="constraint_storage_level_max",
                                         constraint=rules.get_constraint_storage_level_max(),
                                         doc='limit maximum storage level to capacity')
        # couple storage levels
        constraints.add_constraint_block(model, name="constraint_couple_storage_level",
                                         constraint=rules.get_constraint_couple_storage_level(),
                                         doc='couple subsequent storage levels (time coupling constraints)')
        # Linear Capex
        constraints.add_constraint_block(model, name="constraint_storage_technology_capex",
                                         constraint=rules.get_constraint_storage_technology_capex(),
                                         doc='Capital expenditures for installing storage technology')

        # defines disjuncts if technology on/off

    @classmethod
    def disjunct_on_technology_rule(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
        """definition of disjunct constraints if technology is on"""
        params = optimization_setup.parameters
        constraints = optimization_setup.constraints
        model = optimization_setup.model
        energy_system = optimization_setup.energy_system
        # get invest time step
        time_step_year = energy_system.time_steps.convert_time_step_operation2year(tech,time)
        # disjunct constraints min load charge
        constraints.add_constraint_block(model, name=f"constraint_min_load_charge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_charge"][tech, node, time].to_expr()
                                                     - params.min_load.loc[tech, capacity_type, node, time].item() * model.variables["capacity"][tech, capacity_type, node, time_step_year]
                                                     >= 0),
                                         disjunction_var=binary_var)

        # disjunct constraints min load discharge
        constraints.add_constraint_block(model, name=f"constraint_min_load_discharge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_discharge"][tech, node, time].to_expr()
                                                     - params.min_load.loc[tech, capacity_type, node, time].item() * model.variables["capacity"][tech, capacity_type, node, time_step_year]
                                                     >= 0),
                                         disjunction_var=binary_var)

    @classmethod
    def disjunct_off_technology_rule(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
        """definition of disjunct constraints if technology is off"""
        model = optimization_setup.model
        constraints = optimization_setup.constraints

        # for equality constraints we need to add upper and lower bounds
        # off charging
        constraints.add_constraint_block(model, name=f"constraint_off_charging_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_charge"][tech, node, time].to_expr()
                                                     == 0),
                                         disjunction_var=binary_var)

        # off discharging
        constraints.add_constraint_block(model, name=f"constraint_off_discharging_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_discharge"][tech, node, time].to_expr()
                                                     == 0),
                                         disjunction_var=binary_var)


class StorageTechnologyRules:
    """
    Rules for the StorageTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        self.optimization_setup = optimization_setup
        self.energy_system = optimization_setup.energy_system

    ### --- functions with constraint rules --- ###
    def get_constraint_storage_level_max(self):
        """limit maximum storage level to capacity"""
        # get invest time step
        model = self.optimization_setup.model

        # get all the constraints
        constraints = []
        index_values, index_names = Element.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage_level"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        for tech in index.get_unique([0]):
            coords = [model.variables.coords["set_nodes"], model.variables.coords["set_time_steps_storage_level"]]
            nodes, times = index.get_values(locs=[tech], levels=[1, 2], dtype=list, unique=True)
            element_time_step = [self.energy_system.time_steps.convert_time_step_energy2power(tech, t) for t in times]
            time_step_year = [self.energy_system.time_steps.convert_time_step_operation2year(tech, t) for t in element_time_step]
            tuples = [(1.0, model.variables["storage_level"].loc[tech, nodes, times]),
                      (-1.0, model.variables["capacity"].loc[tech, "energy", nodes, time_step_year])]
            constraints.append(linexpr_from_tuple_np(tuples, coords, model)
                               <= 0)

        return self.optimization_setup.constraints.reorder_list(constraints, index.get_unique([0]), index_names[:1], model)

    def get_constraint_couple_storage_level(self):
        """couple subsequent storage levels (time coupling constraints)"""
        # get parameter object
        params = self.optimization_setup.parameters
        system = self.optimization_setup.system
        model = self.optimization_setup.model

        # get all the constraints
        constraints = []
        index_values, index_names = Element.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage_level"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        for tech in index.get_unique([0]):
            nodes, times = index.get_values(locs=[tech], levels=[1, 2], dtype=list, unique=True)
            element_time_step = [self.energy_system.time_steps.convert_time_step_energy2power(tech, t) for t in times]
            # get invest time step
            time_step_year = [self.energy_system.time_steps.convert_time_step_operation2year(tech, t) for t in element_time_step]
            # get corresponding start time step at beginning of the year, if time is last time step in year
            time_step_end = [self.energy_system.time_steps.get_time_steps_storage_startend(tech, t) for t in times]

            # filter end time step and all others
            previous_level_time_step = []
            for t, te in zip(times, time_step_end):
                if te is not None:
                    if system["storage_periodicity"]:
                        previous_level_time_step.append(te)
                    else:
                        previous_level_time_step.append(None)
                else:
                    previous_level_time_step.append(t - 1)
            times = [t for t, tp in zip(times, previous_level_time_step) if tp is not None]
            element_time_step = [t for t, tp in zip(element_time_step, previous_level_time_step) if tp is not None]
            time_step_year = [t for t, tp in zip(time_step_year, previous_level_time_step) if tp is not None]
            previous_level_time_step = [t for t in previous_level_time_step if t is not None]

            # self discharge, reformulate as partial geometric series
            after_self_discharge = xr.where(params.self_discharge.loc[tech, nodes] != 0,
                                            (1-(1 - params.self_discharge.loc[tech, nodes])**params.time_steps_storage_level_duration.loc[tech, times])/(1-(1 - params.self_discharge.loc[tech, nodes])),
                                            params.time_steps_storage_level_duration.loc[tech, times])

            coords = [model.variables.coords["set_nodes"], xr.DataArray(previous_level_time_step, dims=[f"{tech}_{nodes}_set_time_steps_storage_level_end"])]
            tuples = [(1.0, model.variables["storage_level"].loc[tech, nodes, times]),
                      (-(1.0 - params.self_discharge.loc[tech, nodes]) ** params.time_steps_storage_level_duration.loc[tech, times], model.variables["storage_level"].loc[tech, nodes, previous_level_time_step]),
                      (-after_self_discharge.data*params.efficiency_charge.loc[tech, nodes, time_step_year], model.variables["flow_storage_charge"].loc[tech, nodes, element_time_step]),
                      (after_self_discharge.data/params.efficiency_discharge.loc[tech, nodes, time_step_year], model.variables["flow_storage_discharge"].loc[tech, nodes, element_time_step])]
            constraints.append(linexpr_from_tuple_np(tuples, coords, model)
                               == 0)

        return self.optimization_setup.constraints.reorder_list(constraints, index.get_unique([0]), index_names[:1], model)

    def get_constraint_storage_technology_capex(self):
        """ definition of the capital expenditures for the storage technology"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        # get all the constraints
        index_values, index_names = Element.create_custom_set(["set_storage_technologies", "set_capacity_types", "set_nodes", "set_time_steps_yearly"], self.optimization_setup)
        if len(index_values) == 0:
            return []

        techs, capacity_types, nodes, times = IndexSet.tuple_to_arr(index_values, index_names, unique=True)
        coords = [model.variables.coords["set_storage_technologies"], model.variables.coords["set_capacity_types"], model.variables.coords["set_nodes"], model.variables.coords["set_time_steps_yearly"]]
        tuples = [(1.0, model.variables["cost_capex"].loc[techs, capacity_types, nodes, times]),
                  (-params.capex_specific_storage.loc[techs, capacity_types, nodes, times], model.variables["capacity_addition"].loc[techs, capacity_types, nodes, times])]

        return linexpr_from_tuple_np(tuples, coords, model) == 0
