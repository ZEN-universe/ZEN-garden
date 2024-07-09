"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Jacob Mannhardt (jmannhardt@ethz.ch),
               Alissa Ganter (aganter@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the optimization model.
The class takes as inputs the properties of the optimization problem. The properties are saved in the
dictionaries analysis and system which are passed to the class. After initializing the model, the
class adds carriers and technologies to the model and returns it.
The class also includes a method to solve the optimization problem.
"""
import copy
import logging
import os
from collections import defaultdict

import linopy as lp
import numpy as np
import pandas as pd

from .objects.component import Parameter, Variable, Constraint, IndexSet
from .objects.element import Element
from .objects.energy_system import EnergySystem
from .objects.technology.technology import Technology
from zen_garden.preprocess.time_series_aggregation import TimeSeriesAggregation
from zen_garden.preprocess.unit_handling import Scaling

from ..utils import ScenarioDict, IISConstraintParser, StringUtils


class OptimizationSetup(object):
    """setup optimization setup """
    # dict of element classes, this dict is filled in the __init__ of the package
    dict_element_classes = {}

    def __init__(self, config, scenario_dict: dict, input_data_checks):
        """setup Pyomo Concrete Model

        :param config: config object used to extract the analysis, system and solver dictionaries
        :param scenario_dict: dictionary defining the scenario
        :param input_data_checks: input data checks object
        """
        self.analysis = config.analysis
        self.system = config.system
        self.solver = config.solver
        self.input_data_checks = input_data_checks
        self.input_data_checks.optimization_setup = self
        # create a dictionary with the paths to access the model inputs and check if input data exists
        self.create_paths()
        # dict to update elements according to scenario
        self.scenario_dict = ScenarioDict(scenario_dict, config, self.paths)
        # check if all needed data inputs for the chosen technologies exist and remove non-existent
        self.input_data_checks.check_existing_technology_data()
        # empty dict of elements (will be filled with class_name: instance_list)
        self.dict_elements = defaultdict(list)
        # optimization model
        self.model = None
        # the components
        self.variables = None
        self.parameters = None
        self.constraints = None
        self.sets = None


        # sorted list of class names
        element_classes = self.dict_element_classes.keys()
        carrier_classes = [element_name for element_name in element_classes if "Carrier" in element_name]
        technology_classes = [element_name for element_name in element_classes if "Technology" in element_name]
        self.element_list = technology_classes + carrier_classes

        # step of optimization horizon
        self.step_horizon = 0

        # Init the energy system
        self.energy_system = EnergySystem(optimization_setup=self)

        # add Elements to optimization
        self.add_elements()

        # The time series aggregation
        self.time_series_aggregation = None

        # set base scenario
        self.set_base_configuration()

        # read input data into elements
        self.read_input_csv()

        # conduct consistency checks of input units
        self.energy_system.unit_handling.consistency_checks_input_units(optimization_setup=self)

        # conduct time series aggregation
        self.time_series_aggregation = TimeSeriesAggregation(energy_system=self.energy_system)






    def create_paths(self):
        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, technologies
        """
        ## General Paths
        # define path to access dataset related to the current analysis
        self.path_data = self.analysis['dataset']
        assert os.path.exists(self.path_data), f"Folder for input data {self.analysis['dataset']} does not exist!"
        self.input_data_checks.check_primary_folder_structure()
        self.paths = dict()
        # create a dictionary with the keys based on the folders in path_data
        for folder_name in next(os.walk(self.path_data))[1]:
            self.paths[folder_name] = dict()
            self.paths[folder_name]["folder"] = os.path.join(self.path_data, folder_name)
        # add element paths and their file paths
        stack = [self.analysis["subsets"]]
        while stack:
            cur_dict = stack.pop()
            for set_name, subsets in cur_dict.items():
                path = self.paths[set_name]["folder"]
                if isinstance(subsets, dict):
                    stack.append(subsets)
                    self.add_folder_paths(set_name, path, list(subsets.keys()))
                else:
                    self.add_folder_paths(set_name, path, subsets)
                    for element in subsets:
                        if self.system[element]:
                            self.add_folder_paths(element, self.paths[element]["folder"])

    def add_folder_paths(self, set_name, path, subsets=[]):
        """ add file paths of element to paths dictionary
        :param set_name: name of set
        :param path: path to folder
        :param subsets: list of subsets
        """
        for element in next(os.walk(path))[1]:
            if element not in subsets:
                self.paths[set_name][element] = dict()
                self.paths[set_name][element]["folder"] = os.path.join(path, element)
                sub_path = os.path.join(path, element)
                for file in next(os.walk(sub_path))[2]:
                    self.paths[set_name][element][file] = os.path.join(sub_path, file)
                # add element paths to parent sets
                parent_sets = self._find_parent_set(self.analysis["subsets"],set_name)
                for parent_set in parent_sets:
                    self.paths[parent_set][element] = self.paths[set_name][element]
            else:
                self.paths[element] = dict()
                self.paths[element]["folder"] = os.path.join(path, element)

    def _find_parent_set(self,dictionary,subset,path=None):
        """
        This method finds the parent sets of a subset
        :param dictionary: dictionary of subsets
        :param subset: subset to find parent sets of
        :param path: path to subset
        :return: list of parent sets
        """
        if path is None:
            path = []
        for key, value in dictionary.items():
            current_path = path + [key]
            if subset in dictionary[key]:
                return current_path
            elif isinstance(value, dict):
                result = self._find_parent_set(value, subset, current_path)
                if result:
                    return result
        return []

    def add_elements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem."""
        logging.info("\n--- Add elements to model--- \n")
        for element_name in self.element_list:
            element_class = self.dict_element_classes[element_name]
            element_name = element_class.label
            element_set = self.system[element_name]

            # before adding the carriers, get set_carriers and check if carrier data exists
            if element_name == "set_carriers":
                element_set = self.energy_system.set_carriers
                self.system["set_carriers"] = element_set
                self.input_data_checks.check_existing_carrier_data()

            # check if element_set has a subset and remove subset from element_set
            if element_name in self.analysis["subsets"].keys():
                if isinstance(self.analysis["subsets"][element_name], list):
                    subset_names = self.analysis["subsets"][element_name]
                elif isinstance(self.analysis["subsets"][element_name], dict):
                    subset_names = self.analysis["subsets"][element_name].keys()
                else:
                    raise ValueError(f"Subset {element_name} has to be either a list or a dict")
                element_subset = [item for subset in subset_names for item in self.system[subset]]
            else:
                stack = [_dict for _dict in copy.deepcopy(self.analysis["subsets"]).values() if isinstance(_dict, dict)]
                while stack: # check if element_set is a subset of a subset
                    cur_dict = stack.pop()
                    element_subset = []
                    for set_name, subsets in cur_dict.items():
                        if element_name == set_name:
                            if isinstance(subsets, list):
                                element_subset += [item for subset_name in subsets for item in self.system[subset_name]]
                        if isinstance(subsets, dict):
                            stack.append(subsets)
            element_set = list(set(element_set) - set(element_subset))

            element_set.sort()
            # add element class
            for item in element_set:
                self.add_element(element_class, item)

    def read_input_csv(self):
        """ reads the input data of the energy system and elements and conducts the time series aggregation """
        logging.info("\n--- Read input data of elements --- \n")
        self.energy_system.store_input_data()
        for element in self.dict_elements["Element"]:
            element_class = [k for k,v in self.dict_element_classes.items() if v == element.__class__][0]
            logging.info(f"Create {element_class} {element.name}")
            element.store_input_data()
        if self.solver["recommend_base_units"]:
            self.energy_system.unit_handling.recommend_base_units(immutable_unit=self.solver["immutable_unit"],
                                                                  unit_exps=self.solver["range_unit_exponents"])

    def add_element(self, element_class, name):
        """
        Adds an element to the element_dict with the class labels as key

        :param element_class: Class of the element
        :param name: Name of the element
        """
        # get the instance
        instance = element_class(name, self)
        # add to class specific list
        self.dict_elements[element_class.__name__].append(instance)
        # Add the instance to all parents as well
        for cls in element_class.__mro__:
            if not cls == element_class:
                self.dict_elements[cls.__name__].append(instance)

    def get_all_elements(self, cls):
        """ get all elements of the class in the enrgysystem.

        :param cls: class of the elements to return
        :return list of elements in this class """
        return self.dict_elements[cls.__name__]

    def get_all_names_of_elements(self, cls):
        """ get all names of elements in class.

        :param cls: class of the elements to return
        :return names_of_elements: list of names of elements in this class """
        _elements_in_class = self.get_all_elements(cls=cls)
        names_of_elements = []
        for _element in _elements_in_class:
            names_of_elements.append(_element.name)
        return names_of_elements

    def get_element(self, cls, name: str):
        """ get single element in class by name.

        :param name: name of element
        :param cls: class of the elements to return
        :return element: return element whose name is matched """
        for element in self.get_all_elements(cls=cls):
            if element.name == name:
                return element
        return None

    def get_element_class(self, name: str):
        """ get element class by name. If not an element class, return None

        :param name: name of element class
        :return element_class: return element whose name is matched """
        element_classes = {self.dict_element_classes[class_name].label:self.dict_element_classes[class_name] for class_name in self.dict_element_classes}
        if name in element_classes.keys():
            return element_classes[name]
        else:
            return None

    def get_class_set_of_element(self, element_name: str, klass):
        """ returns the set of all elements in the class of the element

        :param element_name: name of element
        :param klass: class of the elements to return
        :return class_set: set of all elements in the class of the element """
        class_name = self.get_element(klass,element_name).__class__.label
        class_set = self.sets[class_name]
        return class_set

    def get_attribute_of_all_elements(self, cls, attribute_name: str, capacity_types=False,
                                      return_attribute_is_series=False):
        """ get attribute values of all elements in a class

        :param cls: class of the elements to return
        :param attribute_name: str name of attribute
        :param capacity_types: boolean if attributes extracted for all capacity types
        :param return_attribute_is_series: boolean if information on attribute type is returned
        :return dict_of_attributes: returns dict of attribute values
        :return attribute_is_series: return information on attribute type """

        class_elements = self.get_all_elements(cls=cls)
        dict_of_attributes = {}
        dict_of_units = {}
        attribute_is_series = False
        for element in class_elements:
            if not capacity_types:
                dict_of_attributes, attribute_is_series_temp, dict_of_units = self.append_attribute_of_element_to_dict(element, attribute_name, dict_of_attributes, dict_of_units)
                if attribute_is_series_temp:
                    attribute_is_series = attribute_is_series_temp
            # if extracted for both capacity types
            else:
                for capacity_type in self.system["set_capacity_types"]:
                    # append energy only for storage technologies
                    if capacity_type == self.system["set_capacity_types"][0] or element.name in self.system["set_storage_technologies"]:
                        dict_of_attributes, attribute_is_series_temp, dict_of_units = self.append_attribute_of_element_to_dict(element, attribute_name, dict_of_attributes, dict_of_units, capacity_type)
                        if attribute_is_series_temp:
                            attribute_is_series = attribute_is_series_temp
        if return_attribute_is_series:
            return dict_of_attributes, dict_of_units, attribute_is_series
        else:
            return dict_of_attributes

    def append_attribute_of_element_to_dict(self, element, attribute_name, dict_of_attributes, dict_of_units, capacity_type=None):
        """ get attribute values of all elements in this class

        :param element: element of class
        :param attribute_name: str name of attribute
        :param dict_of_attributes: dict of attribute values
        :param capacity_type: capacity type for which attribute extracted. If None, not listed in key
        :return dict_of_attributes: returns dict of attribute values """

        attribute_is_series = False
        # add Energy for energy capacity type
        if capacity_type == self.system["set_capacity_types"][1]:
            attribute_name += "_energy"
        # if element does not have attribute
        if not hasattr(element, attribute_name):
            # if attribute is time series that does not exist
            if attribute_name in element.raw_time_series and element.raw_time_series[attribute_name] is None:
                return dict_of_attributes, None, dict_of_units
            else:
                raise AssertionError(f"Element {element.name} does not have attribute {attribute_name}")
        attribute = getattr(element, attribute_name)
        assert not isinstance(attribute, pd.DataFrame), f"Not yet implemented for pd.DataFrames. Wrong format for element {element.name}"
        # add attribute to dict_of_attributes
        if attribute is None:
            return dict_of_attributes, False, dict_of_units
        elif isinstance(attribute, dict):
            dict_of_attributes.update({(element.name,) + (key,): val for key, val in attribute.items()})
        elif isinstance(attribute, pd.Series) and "pwa" not in attribute_name:
            if capacity_type:
                combined_key = (element.name, capacity_type)
            else:
                combined_key = element.name
            if attribute_name in element.units:
                if attribute_name in ["conversion_factor", "retrofit_flow_coupling_factor"]:
                    dict_of_units[combined_key] = element.units[attribute_name]
                else:
                    dict_of_units[combined_key] = element.units[attribute_name]["unit_in_base_units"].units
            else:
                #needed since these
                if attribute_name == "capex_capacity_existing":
                    dict_of_units[combined_key] = element.units["opex_specific_fixed"]["unit_in_base_units"].units
                elif attribute_name == "capex_capacity_existing_energy":
                    dict_of_units[combined_key] = element.units["opex_specific_fixed_energy"]["unit_in_base_units"].units
                elif attribute_name == "capex_specific_transport":
                    dict_of_units[combined_key] = element.units["opex_specific_fixed"]["unit_in_base_units"].units
                elif attribute_name == "capex_per_distance_transport":
                    length_base_unit = [key for key, value in self.energy_system.unit_handling.base_units.items() if value == "[length]"][0]
                    dict_of_units[combined_key] = element.units["opex_specific_fixed"]["unit_in_base_units"].units / self.energy_system.unit_handling.ureg(length_base_unit)
            if len(attribute) > 1:
                dict_of_attributes[combined_key] = attribute
                attribute_is_series = True
            else:
                if attribute.index == 0:
                    dict_of_attributes[combined_key] = attribute.squeeze()
                    attribute_is_series = False
                # since single-directed edges are allowed to exist (e.g. CH-DE exists, DE-CH doesn't), TransportTechnology attributes shared with other technologies (such as capacity existing)
                # mustn't be squeezed even-though the attributes length is smaller than 1. Otherwise, pd.concat(dict_of_attributes) messes up in initialize_component(), leading to an error further on in the code.
                else:
                    dict_of_attributes[combined_key] = attribute
                    attribute_is_series = True
        elif isinstance(attribute, int):
            if capacity_type:
                dict_of_attributes[(element.name, capacity_type)] = [attribute]
            else:
                dict_of_attributes[element.name] = [attribute]
        else:
            if capacity_type:
                dict_of_attributes[(element.name, capacity_type)] = attribute
            else:
                dict_of_attributes[element.name] = attribute
        return dict_of_attributes, attribute_is_series, dict_of_units

    def get_attribute_of_specific_element(self, cls, element_name: str, attribute_name: str):
        """ get attribute of specific element in class

        :param cls: class of the elements to return
        :param element_name: str name of element
        :param attribute_name: str name of attribute
        :return attribute_value: value of attribute"""
        # get element
        _element = self.get_element(cls, element_name)
        # assert that _element exists and has attribute
        assert _element, f"Element {element_name} not in class {cls.__name__}"
        assert hasattr(_element, attribute_name), f"Element {element_name} does not have attribute {attribute_name}"
        attribute_value = getattr(_element, attribute_name)
        return attribute_value

    def construct_optimization_problem(self):
        """ constructs the optimization problem """
        # create empty ConcreteModel
        if self.solver["solver_dir"] is not None and not os.path.exists(self.solver["solver_dir"]):
            os.makedirs(self.solver["solver_dir"])
        self.model = lp.Model(solver_dir=self.solver["solver_dir"])
        # we need to reset the components to not carry them over
        self.sets = IndexSet()
        self.variables = Variable(self)
        self.parameters = Parameter(self)
        self.constraints = Constraint(self.sets,self.model)
        # define and construct components of self.model
        Element.construct_model_components(self)
        # Initiate scaling object
        self.scaling = Scaling(self.model, self.solver['scaling_algorithm'],self.solver['scaling_include_rhs'])
        # find smallest and largest coefficient and RHS
        #self.analyze_numerics() -> Replaced through scaling

    def get_optimization_horizon(self):
        """ returns list of optimization horizon steps """
        # if using rolling horizon
        if self.system.use_rolling_horizon:
            assert self.system.years_in_rolling_horizon >= self.system.interval_between_optimizations, f"There must be more years in the rolling horizon than the interval between optimizations. years_in_rolling_horizon ({self.system.years_in_rolling_horizon}) < interval_between_optimizations ({self.system.interval_between_optimizations})"
            self.years_in_horizon = self.system.years_in_rolling_horizon
            time_steps_yearly = self.energy_system.set_time_steps_yearly
            # skip interval_between_optimizations years
            self.optimized_time_steps = [year for year in time_steps_yearly if (year % self.system.interval_between_optimizations == 0 or year == time_steps_yearly[-1])]
            self.steps_horizon = {year: list(range(year, min(year + self.years_in_horizon, max(time_steps_yearly) + 1))) for year in self.optimized_time_steps}
        # if no rolling horizon
        else:
            self.years_in_horizon = len(self.energy_system.set_time_steps_yearly)
            self.optimized_time_steps = [0]
            self.steps_horizon = {0: self.energy_system.set_time_steps_yearly}
        return list(self.steps_horizon.keys())

    def get_decision_horizon(self,step_horizon):
        """ returns the decision horizon for the optimization step, i.e., the time steps for which the decisions are saved

        :param step_horizon: step of the rolling horizon
        :return decision_horizon: list of time steps in the decision horizon """
        if step_horizon == self.optimized_time_steps[-1]:
            decision_horizon = [step_horizon]
        else:
            next_optimization_step = self.optimized_time_steps[self.optimized_time_steps.index(step_horizon) + 1]
            decision_horizon = list(range(step_horizon, next_optimization_step))
        return decision_horizon

    def set_base_configuration(self, scenario="", elements={}):
        """set base configuration

        :param scenario: name of base scenario
        :param elements: elements in base configuration """
        self.base_scenario = scenario
        self.base_configuration = elements

    def overwrite_time_indices(self, step_horizon):
        """ select subset of time indices, matching the step horizon

        :param step_horizon: step of the rolling horizon """

        if self.system["use_rolling_horizon"]:
            self.step_horizon = step_horizon
            time_steps_yearly_horizon = self.steps_horizon[step_horizon]
            base_time_steps_horizon = self.energy_system.time_steps.decode_yearly_time_steps(time_steps_yearly_horizon)
            # overwrite aggregated time steps - operation
            set_time_steps_operation = self.energy_system.time_steps.encode_time_step(base_time_steps=base_time_steps_horizon,
                                                                                      time_step_type="operation")
            # overwrite aggregated time steps - storage
            set_time_steps_storage = self.energy_system.time_steps.encode_time_step(base_time_steps=base_time_steps_horizon,
                                                                                      time_step_type="storage")
            # copy invest time steps
            time_steps_operation = set_time_steps_operation.squeeze().tolist()
            time_steps_storage = set_time_steps_storage.squeeze().tolist()
            if isinstance(time_steps_operation,int):
                time_steps_operation = [time_steps_operation]
                time_steps_storage = [time_steps_storage]
            self.energy_system.time_steps.time_steps_operation = time_steps_operation
            self.energy_system.time_steps.time_steps_storage = time_steps_storage
            # overwrite base time steps and yearly base time steps
            new_base_time_steps_horizon = base_time_steps_horizon.squeeze().tolist()
            if not isinstance(new_base_time_steps_horizon, list):
                new_base_time_steps_horizon = [new_base_time_steps_horizon]
            self.energy_system.set_base_time_steps = new_base_time_steps_horizon
            self.energy_system.set_time_steps_yearly = time_steps_yearly_horizon

    def solve(self):
        """Create model instance by assigning parameter values and instantiating the sets """
        solver_name = self.solver["name"]
        # remove options that are None
        solver_options = {key: self.solver.solver_options[key] for key in self.solver.solver_options if self.solver.solver_options[key] is not None}

        logging.info(f"\n--- Solve model instance using {solver_name} ---\n")
        # disable logger temporarily
        logging.disable(logging.WARNING)

        if solver_name == "gurobi":
            self.model.solve(solver_name=solver_name, io_api=self.solver["io_api"],
                             keep_files=self.solver["keep_files"], sanitize_zeros=True,
                             # remaining kwargs are passed to the solver
                             **solver_options)
        else:
            self.model.solve(solver_name=solver_name, io_api=self.solver["io_api"],
                             keep_files=self.solver["keep_files"], sanitize_zeros=True)
        # enable logger
        logging.disable(logging.NOTSET)
        if self.model.termination_condition == 'optimal':
            self.optimality = True
        elif self.model.termination_condition == "suboptimal":
            logging.warning("The optimization is suboptimal")
            self.optimality = True
        else:
            self.optimality = False

    def write_IIS(self):
        """ write an ILP file to print the IIS if infeasible. Only possible for gurobi
        """
        if self.model.termination_condition == 'infeasible' and self.solver["name"] == "gurobi":

            output_folder = StringUtils.get_output_folder(self.analysis,self.system)
            ilp_file = os.path.join(output_folder,"infeasible_model_IIS.ilp")
            logging.info(f"Writing parsed IIS to {ilp_file}")
            parser = IISConstraintParser(ilp_file, self.model)
            parser.write_parsed_output()

    def add_results_of_optimization_step(self, step_horizon):
        """ adds the new capacity additions and the cumulative carbon emissions for next
        :param step_horizon: step of the rolling horizon """
        # add newly capacity_addition of first year to existing capacity
        self.add_new_capacity_addition(step_horizon)
        # add cumulative carbon emissions to previous carbon emissions
        self.add_carbon_emission_cumulative(step_horizon)

    def add_new_capacity_addition(self, step_horizon):
        """ adds the newly built capacity to the existing capacity
        :param step_horizon: step of the rolling horizon """
        if self.system["use_rolling_horizon"]:
            if step_horizon != self.energy_system.set_time_steps_yearly_entire_horizon[-1]:
                capacity_addition = self.model.solution["capacity_addition"].to_series().dropna()
                invest_capacity = self.model.solution["capacity_investment"].to_series().dropna()
                cost_capex = self.model.solution["cost_capex"].to_series().dropna()
                if self.solver["round_parameters"]:
                    rounding_value = 10 ** (-self.solver["rounding_decimal_points_capacity"])
                else:
                    rounding_value = 0
                capacity_addition[capacity_addition <= rounding_value] = 0
                invest_capacity[invest_capacity <= rounding_value] = 0
                cost_capex[cost_capex <= rounding_value] = 0
                decision_horizon = self.get_decision_horizon(step_horizon)
                for tech in self.get_all_elements(Technology):
                    # new capacity
                    capacity_addition_tech = capacity_addition.loc[tech.name].unstack()
                    capacity_investment = invest_capacity.loc[tech.name].unstack()
                    cost_capex_tech = cost_capex.loc[tech.name].unstack()
                    tech.add_new_capacity_addition_tech(capacity_addition_tech, cost_capex_tech, decision_horizon)
                    tech.add_new_capacity_investment(capacity_investment, decision_horizon)
            else:
                # TODO clean up
                # reset to initial values
                for tech in self.get_all_elements(Technology):
                    # extract existing capacity
                    set_location = tech.location_type
                    set_time_steps_yearly = self.energy_system.set_time_steps_yearly_entire_horizon
                    self.energy_system.set_time_steps_yearly = copy.deepcopy(set_time_steps_yearly)
                    tech.set_technologies_existing = tech.data_input.extract_set_technologies_existing()
                    tech.capacity_existing = tech.data_input.extract_input_data(
                        "capacity_existing", index_sets=[set_location, "set_technologies_existing"], unit_category={"energy_quantity": 1, "time": -1})
                    tech.capacity_investment_existing = tech.data_input.extract_input_data(
                        "capacity_investment_existing", index_sets=[set_location, "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1, "time": -1})
                    tech.lifetime_existing = tech.data_input.extract_lifetime_existing(
                        "capacity_existing", index_sets=[set_location, "set_technologies_existing"])
                    # calculate capex of existing capacity
                    tech.capex_capacity_existing = tech.calculate_capex_of_capacities_existing()
                    if tech.__class__.__name__ == "StorageTechnology":
                        tech.capacity_existing_energy = tech.data_input.extract_input_data(
                            "capacity_existing_energy", index_sets=["set_nodes", "set_technologies_existing"], unit_category={"energy_quantity": 1})
                        tech.capacity_investment_existing_energy = tech.data_input.extract_input_data(
                            "capacity_investment_existing_energy", index_sets=["set_nodes", "set_time_steps_yearly"],
                            time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1})
                        tech.capex_capacity_existing_energy = tech.calculate_capex_of_capacities_existing(storage_energy=True)

    def add_carbon_emission_cumulative(self, step_horizon):
        """ overwrite previous carbon emissions with cumulative carbon emissions

        :param step_horizon: step of the rolling horizon """
        if self.system["use_rolling_horizon"]:
            if step_horizon != self.energy_system.set_time_steps_yearly_entire_horizon[-1]:
                interval_between_years = self.energy_system.system["interval_between_years"]
                decision_horizon = self.get_decision_horizon(step_horizon)
                last_year = decision_horizon[-1]
                carbon_emissions_cumulative = self.model.solution["carbon_emissions_cumulative"].loc[last_year].item()
                carbon_emissions_annual = self.model.solution["carbon_emissions_annual"].loc[last_year].item()
                self.energy_system.carbon_emissions_cumulative_existing = carbon_emissions_cumulative + carbon_emissions_annual * (interval_between_years - 1)
            else:
                self.energy_system.carbon_emissions_cumulative_existing = self.energy_system.data_input.extract_input_data(
                    "carbon_emissions_cumulative_existing", index_sets=[], unit_category={"emissions": 1})

    def initialize_component(self, calling_class, component_name, index_names=None, set_time_steps=None, capacity_types=False):
        """ this method initializes a modeling component by extracting the stored input data.

        :param calling_class: class from where the method is called
        :param component_name: name of modeling component
        :param index_names: names of index sets, only if calling_class is not EnergySystem
        :param set_time_steps: time steps, only if calling_class is EnergySystem
        :param capacity_types: boolean if extracted for capacities
        :return component_data: data to initialize the component """
        # if calling class is EnergySystem
        if calling_class == EnergySystem:
            component = getattr(self.energy_system, component_name)
            dict_of_units = {}
            if component_name in self.energy_system.units:
                dict_of_units = self.energy_system.units[component_name]
            if index_names is not None:
                index_list = index_names
            elif set_time_steps is not None:
                index_list = [set_time_steps]
            else:
                index_list = []
            if set_time_steps:
                component_data = component[self.sets[set_time_steps]]
            elif type(component) == float:
                component_data = component
            else:
                component_data = component.squeeze()
        else:
            if index_names is None:
                raise ValueError(f"Index names for {component_name} not specified")
            custom_set, index_list = calling_class.create_custom_set(index_names, self)
            component_data, dict_of_units, attribute_is_series = self.get_attribute_of_all_elements(calling_class, component_name, capacity_types=capacity_types, return_attribute_is_series=True)
            if np.size(custom_set):
                if attribute_is_series:
                    component_data = pd.concat(component_data, keys=component_data.keys())
                else:
                    component_data = pd.Series(component_data)
                component_data = self.check_for_subindex(component_data, custom_set)
        if isinstance(component_data,pd.Series) and not isinstance(component_data.index,pd.MultiIndex):
            component_data.index = pd.MultiIndex.from_product([component_data.index.to_list()])
        return component_data, index_list, dict_of_units

    def check_for_subindex(self, component_data, custom_set):
        """ this method checks if the custom_set can be a subindex of component_data and returns subindexed component_data

        :param component_data: extracted data as pd.Series
        :param custom_set: custom set as subindex of component_data
        :return component_data: extracted subindexed data as pd.Series """
        # if custom_set is subindex of component_data, return subset of component_data
        try:
            if len(component_data) == len(custom_set) and len(custom_set[0]) == len(component_data.index[0]):
                return component_data
            else:
                return component_data[custom_set]
        # else delete trivial index levels (that have a single value) and try again
        except:
            _custom_index = pd.Index(custom_set)
            _reduced_custom_index = _custom_index.copy()
            for _level, _shape in enumerate(_custom_index.levshape):
                if _shape == 1:
                    _reduced_custom_index = _reduced_custom_index.droplevel(_level)
            try:
                component_data = component_data[_reduced_custom_index]
                component_data.index = _custom_index
                return component_data
            except KeyError:
                raise KeyError(f"the custom set {custom_set} cannot be used as a subindex of {component_data.index}")

    def analyze_numerics(self):
        """ get largest and smallest matrix coefficients and RHS """
        if self.solver["analyze_numerics"]:
            logging.info("\n--- Analyze Matrix Ranges ---\n")
            flat = self.model.constraints.flat
            # coeffs
            coeffs = flat["coeffs"].abs()
            coeffs = coeffs[coeffs > 0]
            rhs = flat["rhs"].abs()
            rhs = rhs[rhs > 0]
            min_coeff = flat.loc[coeffs.idxmin()]
            max_coeff = flat.loc[coeffs.idxmax()]
            min_rhs = flat.loc[rhs.idxmin()]
            max_rhs = flat.loc[rhs.idxmax()]
            # print coeffs
            max_coeff_str = f"Largest Matrix Coefficient: {self.create_cons_var_string(max_coeff)}"
            min_coeff_str = f"Smallest Matrix Coefficient: {self.create_cons_var_string(min_coeff)}"
            # print rhs
            max_rhs_str = f"Largest RHS: {self.create_cons_var_string(max_rhs,is_coeff=False)}"
            min_rhs_str = f"Smallest RHS: {self.create_cons_var_string(min_rhs,is_coeff=False)}"
            logging.info(f"Numeric Range Statistics:\n{max_coeff_str}\n{min_coeff_str}\n{max_rhs_str}\n{min_rhs_str}")

    def create_cons_var_string(self, cons_series, is_coeff=True):
        """ create a string of constraints or variables

        :param cons_series: pd.Series of constraints or variables
        :param is_coeff: boolean if coefficients, else rhs
        :return cons_str: string of maximum coefficient or rhs"""
        cons_str = self.model.constraints.get_label_position(cons_series["labels"])
        cons_str = cons_str[0] + str(list(cons_str[1].values()))

        if is_coeff:
            var_str = self.model.variables.get_label_position(cons_series["vars"])
            var_str = var_str[0] + str(list(var_str[1].values()))
            coeff_str = abs(cons_series["coeffs"])
            cons_str = f"{coeff_str} {var_str} in {cons_str}"
        else:
            cons_str = f"{abs(cons_series['rhs'])} in {cons_str}"
        return cons_str
