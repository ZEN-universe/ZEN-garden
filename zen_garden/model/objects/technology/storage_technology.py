"""
:Title:          ZEN-GARDEN
:Created:        October-2021
:Authors:        Alissa Ganter (aganter@ethz.ch),
                 Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables and constraints that hold for all storage technologies.
The class takes the abstract optimization model as an input, and returns the parameters, variables and
constraints that hold for the storage technologies.
"""
import cProfile
import logging

import numpy as np
import xarray as xr

from zen_garden.utils import linexpr_from_tuple_np
from .technology import Technology
from ..component import ZenIndex, IndexSet
from ..element import Element, GenericRule


class StorageTechnology(Technology):
    """
    Class defining storage technologies
    """
    # set label
    label = "set_storage_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, optimization_setup):
        """
        init storage technology object

        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(tech, optimization_setup)
        # store carriers of storage technology
        self.store_carriers()

    def store_carriers(self):
        """ retrieves and stores information on reference, input and output carriers """

        # get reference carrier from class <Technology>
        super().store_carriers()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # set attributes for parameters of child class <StorageTechnology>
        self.efficiency_charge = self.data_input.extract_input_data("efficiency_charge", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly")
        self.efficiency_discharge = self.data_input.extract_input_data("efficiency_discharge", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly")
        self.self_discharge = self.data_input.extract_input_data("self_discharge", index_sets=["set_nodes"])
        # extract existing energy capacity
        self.capacity_addition_min_energy = self.data_input.extract_input_data("capacity_addition_min_energy", index_sets=[])
        self.capacity_addition_max_energy = self.data_input.extract_input_data("capacity_addition_max_energy", index_sets=[])
        self.capacity_limit_energy = self.data_input.extract_input_data("capacity_limit_energy", index_sets=["set_nodes"])
        self.capacity_existing_energy = self.data_input.extract_input_data("capacity_existing_energy", index_sets=["set_nodes", "set_technologies_existing"])
        self.capacity_investment_existing_energy = self.data_input.extract_input_data("capacity_investment_existing_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly")
        self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly")
        self.capex_specific_energy = self.data_input.extract_input_data("capex_specific_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly")
        self.opex_specific_fixed_energy = self.data_input.extract_input_data("opex_specific_fixed_energy", index_sets=["set_nodes", "set_time_steps_yearly"],
                                                                            time_steps="set_time_steps_yearly")
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()
        self.capex_capacity_existing_energy = self.calculate_capex_of_capacities_existing(storage_energy=True)
        # add min load max load time series for energy
        self.raw_time_series["min_load_energy"] = self.data_input.extract_input_data("min_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly")
        self.raw_time_series["max_load_energy"] = self.data_input.extract_input_data("max_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly")

    def convert_to_fraction_of_capex(self):
        """ this method converts the total capex to fraction of capex, depending on how many hours per year are calculated """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.opex_specific_fixed_energy = self.opex_specific_fixed_energy * fraction_year
        self.capex_specific = self.capex_specific * fraction_year
        self.capex_specific_energy = self.capex_specific_energy * fraction_year

    def calculate_capex_of_single_capacity(self, capacity, index, storage_energy=False):
        """ this method calculates the annualized capex of a single existing capacity.

        :param capacity: capacity of storage technology
        :param index: index of capacity
        :param storage_energy: boolean if energy capacity or power capacity
        :return: capex of single capacity
        """
        if storage_energy:
            absolute_capex = self.capex_specific_energy[index[0]].iloc[0] * capacity
        else:
            absolute_capex = self.capex_specific[index[0]].iloc[0] * capacity
        return absolute_capex

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to StorageTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <StorageTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        pass

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <StorageTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
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
            time_step_year = xr.DataArray([optimization_setup.energy_system.time_steps.convert_time_step_operation2year(time) for time in time_arr.data])
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
        variables.add_variable(model, name="storage_level", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage"], optimization_setup), bounds=(0, np.inf),
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
                                         constraint=rules.constraint_storage_level_max_block(),
                                         doc='limit maximum storage level to capacity')
        # couple storage levels
        constraints.add_constraint_block(model, name="constraint_couple_storage_level",
                                         constraint=rules.constraint_couple_storage_level_block(),
                                         doc='couple subsequent storage levels (time coupling constraints)')
        # Linear Capex
        constraints.add_constraint_block(model, name="constraint_storage_technology_capex",
                                         constraint=rules.constraint_storage_technology_capex_block(),
                                         doc='Capital expenditures for installing storage technology')

        # defines disjuncts if technology on/off

    @classmethod
    def disjunct_on_technology_rule(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
        """definition of disjunct constraints if technology is on

        :param optimization_setup: optimization setup
        :param tech: technology
        :param capacity_type: type of capacity (power, energy)
        :param node: node
        :param time: yearly time step
        :param binary_var: binary disjunction variable
        """
        params = optimization_setup.parameters
        constraints = optimization_setup.constraints
        model = optimization_setup.model
        energy_system = optimization_setup.energy_system
        # get invest time step
        time_step_year = energy_system.time_steps.convert_time_step_operation2year(tech,time)
        # disjunct constraints min load charge
        constraints.add_constraint_block(model, name=f"disjunct_storage_technology_min_load_charge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_charge"][tech, node, time].to_expr()
                                                     - params.min_load.loc[tech, capacity_type, node, time].item() * model.variables["capacity"][tech, capacity_type, node, time_step_year]
                                                     >= 0),
                                         disjunction_var=binary_var)

        # disjunct constraints min load discharge
        constraints.add_constraint_block(model, name=f"disjunct_storage_technology_min_load_discharge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_discharge"][tech, node, time].to_expr()
                                                     - params.min_load.loc[tech, capacity_type, node, time].item() * model.variables["capacity"][tech, capacity_type, node, time_step_year]
                                                     >= 0),
                                         disjunction_var=binary_var)

    @classmethod
    def disjunct_off_technology_rule(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
        """definition of disjunct constraints if technology is off

        :param optimization_setup: optimization setup
        :param tech: technology
        :param capacity_type: type of capacity (power, energy)
        :param node: node
        :param time: yearly time step
        :param binary_var: binary disjunction variable
        """
        model = optimization_setup.model
        constraints = optimization_setup.constraints

        # for equality constraints we need to add upper and lower bounds
        # off charging
        constraints.add_constraint_block(model, name=f"disjunct_storage_technology_off_charge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_charge"][tech, node, time].to_expr()
                                                     == 0),
                                         disjunction_var=binary_var)

        # off discharging
        constraints.add_constraint_block(model, name=f"disjunct_storage_technology_off_discharge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_discharge"][tech, node, time].to_expr()
                                                     == 0),
                                         disjunction_var=binary_var)


class StorageTechnologyRules(GenericRule):
    """
    Rules for the StorageTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem

        :param optimization_setup: The OptimizationSetup the element is part of
        """

        super().__init__(optimization_setup)

    # Rule-based constraints
    # ----------------------

    # Block-based constraints
    # -----------------------

    def constraint_storage_level_max_block(self):
        """limit maximum storage level to capacity

        .. math::
            L_{k,n,t^\mathrm{k}} \le S^\mathrm{e}_{k,n,y}

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over the technologies for the time step conversion
        # we vectorize over the nodes and storage time steps
        constraints = []
        for tech in index.get_unique(["set_storage_technologies"]):

            ### auxiliary calculations
            coords = [self.variables.coords["set_nodes"], self.variables.coords["set_time_steps_storage"]]
            nodes, times = index.get_values(locs=[tech], levels=[1, 2], dtype=list, unique=True)
            element_time_step = [self.energy_system.time_steps.convert_time_step_energy2power(t) for t in times]
            time_step_year = [self.energy_system.time_steps.convert_time_step_operation2year(t) for t in element_time_step]

            ### formulate constraint
            lhs = linexpr_from_tuple_np([(1.0, self.variables["storage_level"].loc[tech, nodes, times]),
                                         (-1.0, self.variables["capacity"].loc[tech, "energy", nodes, time_step_year])],
                                        coords, self.model)
            rhs = 0
            constraints.append(lhs <= rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(levels=["set_storage_technologies"]),
                                                  index_names=["set_storage_technologies"])

    def constraint_couple_storage_level_block(self):
        """couple subsequent storage levels (time coupling constraints)

        .. math::
            L(t) = L_0\\kappa^t + \\Delta H\\frac{1-\\kappa^t}{1-\\kappa} = \\frac{\\Delta H}{1-\\kappa}+(L_0-\\frac{\\Delta H}{1-\\kappa})\\kappa^t

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over the technologies for the time step conversion
        # we vectorize over the nodes and storage time steps
        constraints = []
        for tech in index.get_unique(["set_storage_technologies"]):

            ### auxiliary calculations
            nodes, times = index.get_values(locs=[tech], levels=[1, 2], dtype=list, unique=True)
            element_time_step = [self.energy_system.time_steps.convert_time_step_energy2power(t) for t in times]
            # get invest time step
            time_step_year = [self.energy_system.time_steps.convert_time_step_operation2year(t) for t in
                              element_time_step]
            # get corresponding start time step at beginning of the year, if time is last time step in year
            time_step_end = [self.energy_system.time_steps.get_time_steps_storage_startend(t) for t in times]

            # filter end time step and all others
            previous_level_time_step = []
            for t, te in zip(times, time_step_end):
                if te is not None:
                    if self.system["storage_periodicity"]:
                        previous_level_time_step.append(te)
                    else:
                        previous_level_time_step.append(None)
                else:
                    previous_level_time_step.append(self.energy_system.time_steps.get_previous_storage_time_step(t))
            times = [t for t, tp in zip(times, previous_level_time_step) if tp is not None]
            element_time_step = [t for t, tp in zip(element_time_step, previous_level_time_step) if tp is not None]
            time_step_year = [t for t, tp in zip(time_step_year, previous_level_time_step) if tp is not None]
            previous_level_time_step = [t for t in previous_level_time_step if t is not None]

            # self discharge, reformulate as partial geometric series
            after_self_discharge = xr.where(self.parameters.self_discharge.loc[tech, nodes] != 0,
                                            (1 - (1 - self.parameters.self_discharge.loc[tech, nodes]) **
                                             self.parameters.time_steps_storage_duration.loc[times]) / (
                                                        1 - (1 - self.parameters.self_discharge.loc[tech, nodes])),
                                            self.parameters.time_steps_storage_duration.loc[times])

            coords = [self.variables.coords["set_nodes"],
                      xr.DataArray(previous_level_time_step, dims=[f"{tech}_{nodes}_set_time_steps_storage_end"])]

            ### formulate constraint
            lhs = linexpr_from_tuple_np([(1.0, self.variables["storage_level"].loc[tech, nodes, times]),
                                         (-(1.0 - self.parameters.self_discharge.loc[tech, nodes]) ** self.parameters.time_steps_storage_duration.loc[times], self.variables["storage_level"].loc[tech, nodes, previous_level_time_step]),
                                         (-after_self_discharge.data*self.parameters.efficiency_charge.loc[tech, nodes, time_step_year], self.variables["flow_storage_charge"].loc[tech, nodes, element_time_step]),
                                         (after_self_discharge.data/self.parameters.efficiency_discharge.loc[tech, nodes, time_step_year], self.variables["flow_storage_discharge"].loc[tech, nodes, element_time_step])],
                                        coords, self.model)
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_storage_technologies"]),
                                                  index_names=["set_storage_technologies"])

    def constraint_storage_technology_capex_block(self):
        """ definition of the capital expenditures for the storage technology

        .. math::
            CAPEX_{y,n,i}^\mathrm{cost} = \\Delta S_{h,p,y} \\alpha_{k,n,y}

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_storage_technologies", "set_capacity_types", "set_nodes", "set_time_steps_yearly"], self.optimization_setup)
        # check if we need to continue
        if len(index_values) == 0:
            return []

        ### masks
        # not necessary

        ### index loop
        # not necessary

        ### auxiliary calculations
        # get all the arrays and coords
        techs, capacity_types, nodes, times = IndexSet.tuple_to_arr(index_values, index_names, unique=True)
        coords = [self.variables.coords["set_storage_technologies"], self.variables.coords["set_capacity_types"], self.variables.coords["set_nodes"], self.variables.coords["set_time_steps_yearly"]]

        ### formulate constraint
        lhs = linexpr_from_tuple_np([(1.0, self.variables["cost_capex"].loc[techs, capacity_types, nodes, times]),
                                     (-self.parameters.capex_specific_storage.loc[techs, capacity_types, nodes, times], self.variables["capacity_addition"].loc[techs, capacity_types, nodes, times])],
                                     coords, self.model)
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)
