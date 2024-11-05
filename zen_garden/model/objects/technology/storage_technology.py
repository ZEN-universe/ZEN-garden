"""
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
        self.efficiency_charge = self.data_input.extract_input_data("efficiency_charge", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={})
        self.efficiency_discharge = self.data_input.extract_input_data("efficiency_discharge", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={})
        self.self_discharge = self.data_input.extract_input_data("self_discharge", index_sets=["set_nodes"], unit_category={})
        # extract existing energy capacity
        self.capacity_addition_min_energy = self.data_input.extract_input_data("capacity_addition_min_energy", index_sets=[], unit_category={"energy_quantity": 1})
        self.capacity_addition_max_energy = self.data_input.extract_input_data("capacity_addition_max_energy", index_sets=[], unit_category={"energy_quantity": 1})
        self.capacity_limit_energy = self.data_input.extract_input_data("capacity_limit_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1})
        self.capacity_existing_energy = self.data_input.extract_input_data("capacity_existing_energy", index_sets=["set_nodes", "set_technologies_existing"], unit_category={"energy_quantity": 1})
        self.capacity_investment_existing_energy = self.data_input.extract_input_data("capacity_investment_existing_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1})
        self.energy_to_power_ratio_min = self.data_input.extract_input_data("energy_to_power_ratio_min", index_sets=[], unit_category={"time": 1})
        self.energy_to_power_ratio_max = self.data_input.extract_input_data("energy_to_power_ratio_max", index_sets=[],
                                                                        unit_category={"time": 1})
        self.capex_specific_storage = self.data_input.extract_input_data("capex_specific_storage", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1, "time": -1})
        self.capex_specific_storage_energy = self.data_input.extract_input_data("capex_specific_storage_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1})
        self.opex_specific_fixed = self.data_input.extract_input_data("opex_specific_fixed", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1, "time": 1})
        self.opex_specific_fixed_energy = self.data_input.extract_input_data("opex_specific_fixed_energy", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1})
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()
        self.capex_capacity_existing_energy = self.calculate_capex_of_capacities_existing(storage_energy=True)
        # add min load max load time series for energy
        self.raw_time_series["min_load_energy"] = self.data_input.extract_input_data("min_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={})
        self.raw_time_series["max_load_energy"] = self.data_input.extract_input_data("max_load_energy", index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={})
        # add flow_storage_inflow time series
        self.raw_time_series["flow_storage_inflow"] = self.data_input.extract_input_data("flow_storage_inflow",index_sets=["set_nodes", "set_time_steps"],time_steps="set_base_time_steps_yearly",unit_category={"energy_quantity": 1, "time": -1})

    def convert_to_fraction_of_capex(self):
        """ this method converts the total capex to fraction of capex, depending on how many hours per year are calculated """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.opex_specific_fixed_energy = self.opex_specific_fixed_energy * fraction_year
        self.capex_specific_storage = self.capex_specific_storage * fraction_year
        self.capex_specific_storage_energy = self.capex_specific_storage_energy * fraction_year

    def calculate_capex_of_single_capacity(self, capacity, index, storage_energy=False):
        """ this method calculates the annualized capex of a single existing capacity.

        :param capacity: capacity of storage technology
        :param index: index of capacity
        :param storage_energy: boolean if energy capacity or power capacity
        :return: capex of single capacity
        """
        if storage_energy:
            absolute_capex = self.capex_specific_storage_energy[index[0]].iloc[0] * capacity
        else:
            absolute_capex = self.capex_specific_storage[index[0]].iloc[0] * capacity
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
        # energy to power ratio
        optimization_setup.parameters.add_parameter(name="energy_to_power_ratio_min", index_names=["set_storage_technologies"], doc='power to energy ratio for storage technologies - lower bound', calling_class=cls)
        optimization_setup.parameters.add_parameter(name="energy_to_power_ratio_max", index_names=["set_storage_technologies"], doc='power to energy ratio for storage technologies - upper bound', calling_class=cls)
        # efficiency charge
        optimization_setup.parameters.add_parameter(name="efficiency_charge", index_names=["set_storage_technologies", "set_nodes", "set_time_steps_yearly"], doc='efficiency during charging for storage technologies', calling_class=cls)
        # efficiency discharge
        optimization_setup.parameters.add_parameter(name="efficiency_discharge", index_names=["set_storage_technologies", "set_nodes", "set_time_steps_yearly"], doc='efficiency during discharging for storage technologies', calling_class=cls)
        #  flow_storage_inflow
        optimization_setup.parameters.add_parameter(name="flow_storage_inflow", index_names=["set_storage_technologies", "set_nodes", "set_time_steps_operation"], doc='energy inflow in storage technologies', calling_class=cls)
        # self discharge
        optimization_setup.parameters.add_parameter(name="self_discharge", index_names=["set_storage_technologies", "set_nodes"], doc='self discharge of storage technologies', calling_class=cls)
        # capex specific
        optimization_setup.parameters.add_parameter(name="capex_specific_storage", index_names=["set_storage_technologies", "set_capacity_types", "set_nodes", "set_time_steps_yearly"], capacity_types=True, doc='specific capex of storage technologies', calling_class=cls)

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
            bounds=bounds, doc='carrier flow into storage technology on node i and time t', unit_category={"energy_quantity": 1, "time": -1})
        # flow of carrier on node out of storage
        variables.add_variable(model, name="flow_storage_discharge", index_sets=(index_values, index_names),
            bounds=bounds, doc='carrier flow out of storage technology on node i and time t', unit_category={"energy_quantity": 1, "time": -1})
        # storage level
        variables.add_variable(model, name="storage_level", index_sets=cls.create_custom_set(["set_storage_technologies", "set_nodes", "set_time_steps_storage"], optimization_setup), bounds=(0, np.inf),
            doc='storage level of storage technology ón node in each storage time step', unit_category={"energy_quantity": 1})
        # energy spillage
        variables.add_variable(model, name="flow_storage_spillage", index_sets=(index_values, index_names), bounds=(0, np.inf), doc='storage spillage of storage technology on node i in each storage time step', unit_category={"energy_quantity": 1, "time": -1})

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the Constraints of the class <StorageTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        rules = StorageTechnologyRules(optimization_setup)
        # limit flow by capacity and max load
        rules.constraint_capacity_factor_storage()

        # opex and emissions constraint for storage technologies
        rules.constraint_opex_emissions_technology_storage()

        # Limit storage level
        rules.constraint_storage_level_max()

        # couple storage levels
        rules.constraint_couple_storage_level()

        # spillage limit
        rules.constraint_flow_storage_spillage()

        # limit energy to power ratios of capacity additions
        rules.constraint_capacity_energy_to_power_ratio()

        # Linear Capex
        rules.constraint_storage_technology_capex()

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
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
        # TODO make to constraint rule or integrate in new structure!!!
        time_step_year = energy_system.time_steps.convert_time_step_operation2year(time)
        # get min load limit
        min_load = params.min_load.loc[tech, capacity_type, node, time] * model.variables["capacity"].loc[tech, capacity_type, node, time_step_year]
        # formulate constraint
        lhs = model.variables["flow_storage_charge"].loc[tech, node, time] - min_load
        rhs = 0
        constraint = lhs >= rhs
        # disjunct constraints min load charge
        constraints.add_constraint_block(model, name=f"disjunct_storage_technology_min_load_charge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint= constraint, disjunction_var=binary_var)
        # formulate constraint
        lhs =  model.variables["flow_storage_discharge"].loc[tech, node, time] - min_load
        rhs = 0
        constraint = lhs >= rhs
        # disjunct constraints min load discharge
        constraints.add_constraint_block(model, name=f"disjunct_storage_technology_min_load_discharge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=constraint, disjunction_var=binary_var)

    @classmethod
    def disjunct_off_technology(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
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
                                         constraint=(model.variables["flow_storage_charge"].loc[tech, node, time]
                                                     == 0),
                                         disjunction_var=binary_var)

        # off discharging
        constraints.add_constraint_block(model, name=f"disjunct_storage_technology_off_discharge_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=(model.variables["flow_storage_discharge"].loc[tech, node, time]
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

    def constraint_capacity_factor_storage(self):
        """ Load is limited by the installed capacity and the maximum load factor for storage technologies

        .. math::
            \\underline{H}_{k,n,t,y}+\\overline{H}_{k,n,t,y}\\leq m^{\mathrm{max}}_{k,n,t,y}S_{k,n,y}

        :math:`\\underline{H}_{k,n,t,y}`: carrier flow into storage technology :math:`k` on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`\\overline{H}_{k,n,t,y}`: carrier flow out of storage technology :math:`k`on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`m^{\mathrm{max}}_{k,n,t,y}`: maximum load factor for storage technology :math:`k` on node :math:`n` and time :math:`t` in year :math:`y` \n
        :math:`S_{k,n,y}`: storage capacity of storage technology :math:`k` on node :math:`n` in year :math:`y`


        """
        techs = self.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = self.sets["set_nodes"]
        times = self.variables.coords["set_time_steps_operation"]
        time_step_year = xr.DataArray([self.optimization_setup.energy_system.time_steps.convert_time_step_operation2year(t) for t in times.data], coords=[times])
        term_capacity = (
                self.parameters.max_load.loc[techs, "power", nodes, :]
                * self.variables["capacity"].loc[techs, "power", nodes, time_step_year]
        ).rename({"set_technologies": "set_storage_technologies","set_location":"set_nodes"})

        # TODO integrate level storage here as well
        lhs = term_capacity - self.get_flow_expression_storage(rename=False)
        rhs = 0
        constraints = lhs >= rhs
        ### return
        self.constraints.add_constraint("constraint_capacity_factor_storage",constraints)

    def constraint_opex_emissions_technology_storage(self):
        """ calculate opex of each technology

        .. math::
            O_{h,p,t}^\mathrm{t} = \\beta_{h,p,t} (\\underline{H}_{k,n,t} + \\overline{H}_{k,n,t}) \n
            \\theta_{h,p,t}^{\mathrm{tech}} = \\epsilon_h (\\underline{H}_{k,n,t} + \\overline{H}_{k,n,t})

        :math:`O_{h,p,t}^\mathrm{t}`: variable operational expenditures for storage technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\beta_{h,p,t}`: specific variable operational expenditures for storage technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\underline{H}_{k,n,t}`: carrier flow into storage technology :math:`k` on node :math:`n` and time :math:`t` \n
        :math:`\\overline{H}_{k,n,t}`: carrier flow out of storage technology :math:`k` on node :math:`n` and time :math:`t` \n
        :math:`\\theta_{h,p,t}^{\mathrm{tech}}`: carbon emissions for storage technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\epsilon_h`: carbon intensity for operating storage technology :math:`h` on node :math:`n`





        """
        techs = self.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = self.sets["set_nodes"]
        lhs_opex = (
                self.variables["cost_opex_variable"] - (self.parameters.opex_specific_variable * self.get_flow_expression_storage())
        ).sel({"set_technologies":techs,"set_location":nodes})
        lhs_emissions = (self.variables["carbon_emissions_technology"]
               - (self.parameters.carbon_intensity_technology*self.get_flow_expression_storage())).sel({"set_technologies":techs,"set_location":nodes})
        lhs_opex = lhs_opex.rename({"set_technologies": "set_storage_technologies","set_location":"set_nodes"})
        lhs_emissions = lhs_emissions.rename({"set_technologies": "set_storage_technologies","set_location":"set_nodes"})
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs

        self.constraints.add_constraint("constraint_opex_technology_storage",constraints_opex)
        self.constraints.add_constraint("constraint_carbon_emissions_technology_storage",constraints_emissions)

    def constraint_storage_level_max(self):
        """limit maximum storage level to capacity

        .. math::
            L_{k,n,t^\mathrm{k}} \le S^\mathrm{e}_{k,n,y}

        :math:`L_{k,n,t^\mathrm{k}}`: storage level of storage technology :math:`k` on node :math:`n` and time :math:`t` \n
        :math:`S^\mathrm{e}_{k,n,y}`: energy capacity of storage technology :math:`k` on node :math:`n` in year :math:`y`

        """
        techs = self.sets["set_storage_technologies"]
        nodes = self.sets["set_nodes"]
        if len(techs) == 0:
            return
        # mask for energy capacity and storage time steps
        times = self.get_storage2year_time_step_array()
        capacity = self.map_and_expand(self.variables["capacity"],times)
        capacity = capacity.rename({"set_technologies": "set_storage_technologies","set_location":"set_nodes"})
        capacity = capacity.sel({"set_nodes":nodes,"set_storage_technologies":techs})
        storage_level = self.variables["storage_level"]
        mask_capacity_type = self.variables["capacity"].coords["set_capacity_types"] == "energy"
        lhs = (storage_level - capacity).where(mask_capacity_type,0.0)
        rhs = 0
        constraints = lhs <= rhs

        self.constraints.add_constraint("constraint_storage_level_max",constraints)

    def constraint_capacity_energy_to_power_ratio(self):
        """limit capacity power to energy ratio

        .. math::
            \\rho_k^{min} S^{e}_{k,n,y} \le S_{k,n,y}

        .. math::
            S_{k,n,y} \le \\rho_k^{max} S^{e}_{k,n,y}

        :math:`S^{\mathrm{power}}_{k,n,y}`: installed capacity in terms of power of storage :math:`k` at node :math:`n` in year :math:`y` \n
        :math:`S^{e}_{k,n,y}`: installed capacity in terms of energy of storage :math:`k` at node :math:`n` in year :math:`y` \n
        :math:`\\rho_k^{min}`: minimum power-to-energy ratio of storage :math:`k` \n
        :math:`\\rho_k^{max}`: maximum power-to-energy ratio of storage :math:`k`

        """



        techs = self.sets["set_storage_technologies"]
        if len(techs) == 0:
            return None
        e2p_min = self.parameters.energy_to_power_ratio_min
        e2p_max = self.parameters.energy_to_power_ratio_max
        mask_min = e2p_min != np.inf
        mask_max = e2p_max != np.inf

        capacity_addition = self.variables["capacity_addition"].rename({"set_technologies": "set_storage_technologies"})
        capacity_addition_power = capacity_addition.sel({"set_storage_technologies":techs,"set_capacity_types": "power"})
        capacity_addition_energy = capacity_addition.sel({"set_storage_technologies":techs,"set_capacity_types": "energy"})
        lhs = (capacity_addition_energy - capacity_addition_power * e2p_min).where(mask_min)
        rhs = 0
        constraints_min = lhs >= rhs
        lhs = (capacity_addition_energy - capacity_addition_power*e2p_max).where(mask_max)
        constraints_max = lhs <= rhs

        self.constraints.add_constraint("constraint_capacity_energy_to_power_ratio_min", constraints_min)
        self.constraints.add_constraint("constraint_capacity_energy_to_power_ratio_max", constraints_max)

    def constraint_couple_storage_level(self):
        """couple subsequent storage levels (time coupling constraints)

        .. math::
            L_{k,n,t^k,y} = L_{k,n,t^k-1,y} (1-\\phi_k)^{\\tau_{t^k}^k} + (\\underline{\\eta}_k \\underline{H}_{k,n,\\sigma(t^k),y} - \\frac{\\overline{H}_{k,n,\\sigma(t^k),y}}{\\overline{\\eta}_k}) \sum^{\\tau_{t^k}^k-1}_{\\tilde{t}^k=0} (1-\\phi_k)^{\\tilde{t}^k}

        :math:`L_{k,n,t^k,y}`: storage level of storage technology :math:`k` on node :math:`n` and time :math:`t^k` in year :math:`y` \n
        :math:`\\phi_k`: self discharge rate of storage technology :math:`k` \n
        :math:`\\tau_{t^k}^k`: duration of storage level time step of storage technology :math:`k` \n
        :math:`\\underline{\\eta}_k`: efficiency during charging of storage technology :math:`k` \n
        :math:`\\overline{\\eta}_k`: efficiency during discharging of storage technology :math:`k` \n
        :math:`\\underline{H}_{k,n,\\sigma(t^k),y}`: charge flow into storage technology :math:`k` on node :math:`n` and time :math:`\\sigma(t^k)` in year :math:`y` \n
        :math:`\\overline{H}_{k,n,\\sigma(t^k),y}`: discharge flow out of storage technology :math:`k` on node :math:`n` and time :math:`\\sigma(t^k)` in year :math:`y`

        """
        techs = self.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        self_discharge = self.parameters.self_discharge
        flow_storage_inflow = self.parameters.flow_storage_inflow
        flow_storage_spillage = self.variables.flow_storage_spillage
        time_steps_storage_duration = self.parameters.time_steps_storage_duration
        # reformulate self discharge multiplier as partial geometric series
        multiplier_w_discharge = (
                    (1 - (1 - self_discharge) ** (time_steps_storage_duration)) / (1 - (1 - self_discharge)))
        multiplier_wo_discharge = time_steps_storage_duration
        multiplier = (
                multiplier_w_discharge.where(self_discharge != 0, 0.0) +
                multiplier_wo_discharge.where(self_discharge == 0, 0.0))
        # time coupling to previous time step
        times_coupling,mask_coupling = self.get_previous_storage_time_step_array()
        self_discharge_previous = (1-self_discharge)**time_steps_storage_duration
        self_discharge_previous["set_time_steps_storage"] = times_coupling
        term_delta_storage_level = (
                self.variables["storage_level"] - self_discharge_previous * self.variables["storage_level"].sel({"set_time_steps_storage": times_coupling}))
        # charge and discharge flow
        times_year_time_step = self.get_year_time_step_array()
        efficiency_charge = self.parameters.efficiency_charge.broadcast_like(times_year_time_step).where(times_year_time_step,0.0).sum("set_time_steps_yearly")
        efficiency_discharge = self.parameters.efficiency_discharge.broadcast_like(times_year_time_step).where(times_year_time_step,0.0).sum("set_time_steps_yearly")
        term_flow_charge_discharge = (
                self.variables["flow_storage_charge"] * efficiency_charge -
                self.variables["flow_storage_discharge"] / efficiency_discharge +
                flow_storage_inflow -
                flow_storage_spillage)
        times_power2energy = self.get_power2energy_time_step_array()
        term_flow_charge_discharge = self.map_and_expand(term_flow_charge_discharge, times_power2energy)
        term_flow_charge_discharge = term_flow_charge_discharge*multiplier
        # sum up all terms
        lhs = (term_delta_storage_level - term_flow_charge_discharge).where(mask_coupling,0.0)
        rhs = 0
        constraints = lhs == rhs

        self.constraints.add_constraint("constraint_couple_storage_level",constraints)

    def constraint_flow_storage_spillage(self):
        """Impose that the flow_energy_spillage cannot be greater than the flow_storage_inflow.
        .. math::
            TODO
        """
        techs = self.sets["set_storage_technologies"]
        if len(techs) == 0:
            return

        flow_storage_inflow = self.parameters.flow_storage_inflow
        flow_storage_spillage = self.variables.flow_storage_spillage

        lhs = flow_storage_spillage - flow_storage_inflow
        rhs = 0
        constraints = lhs <= rhs

        self.constraints.add_constraint("constraint_flow_storage_spillage", constraints)

    def constraint_storage_technology_capex(self):
        """ definition of the capital expenditures for the storage technology

        .. math::
            CAPEX_{y,n,i} = \\Delta S_{h,p,y} \\alpha_{k,n,y}

        :math:`\\Delta S_{h,p,y}`: capacity addition of storage technology :math:`h` on node :math:`n` in year :math:`y` \n
        :math:`\\alpha_{k,n,y}`: specific capex of storage technology :math:`k` on node :math:`n` in year :math:`y`


        """
        # TODO clean up
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
        lhs = linexpr_from_tuple_np([(1.0, self.variables["cost_capex_overnight"].loc[techs, capacity_types, nodes, times]),
                                     (-self.parameters.capex_specific_storage.loc[techs, capacity_types, nodes, times], self.variables["capacity_addition"].loc[techs, capacity_types, nodes, times])],
                                     coords, self.model)
        rhs = 0
        constraints = lhs == rhs

        self.constraints.add_constraint("constraint_storage_technology_capex",constraints)

    def get_flow_expression_storage(self,rename=True):
        """ return the flow expression for storage technologies """
        term = (self.variables["flow_storage_charge"] + self.variables["flow_storage_discharge"])
        if rename:
            return term.rename({"set_storage_technologies": "set_technologies","set_nodes":"set_location"})
        else:
            return term
