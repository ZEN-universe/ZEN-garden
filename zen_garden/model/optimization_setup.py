"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class defining the Concrete optimization model.
              The class takes as inputs the properties of the optimization problem. The properties are saved in the
              dictionaries analysis and system which are passed to the class. After initializing the Concrete model, the
              class adds carriers and technologies to the Concrete model and returns the Concrete optimization model.
              The class also includes a method to solve the optimization problem.
==========================================================================================================================================================================="""
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pyomo.environ as pe
from pyomo.core.expr.current import decompose_term

from .objects.element import Element
from .objects.energy_system import EnergySystem
from .objects.component import Parameter, Variable, Constraint
from .objects.technology.technology import Technology
from ..preprocess.functions.time_series_aggregation import TimeSeriesAggregation
from ..preprocess.prepare import Prepare


class OptimizationSetup(object):

    # dict of element classes, this dict is filled in the __init__ of the package
    dict_element_classes = {}

    def __init__(self, analysis: dict, prepare: Prepare, energy_system_name="energy_system"):
        """setup Pyomo Concrete Model
        :param analysis: dictionary defining the analysis framework
        :param prepare: A Prepare instance for the Optimization setup"""

        self.prepare = prepare
        self.analysis = analysis
        self.system = prepare.system
        self.paths = prepare.paths
        self.solver = prepare.solver

        # empty dict of elements (will be filled with class_name: instance_list)
        self.dict_elements = defaultdict(list)
        # pe.ConcreteModel
        self.pyomo_model = None
        # the objectives
        self.objectives = Objectives(self)
        # the components
        self.variables = None
        self.parameters = None
        self.constraints = None

        # sorted list of class names
        element_classes = self.dict_element_classes.keys()
        carrier_classes = [element_name for element_name in element_classes if "Carrier" in element_name]
        technology_classes = [element_name for element_name in element_classes if "Technology" in element_name]
        self.element_list = technology_classes + carrier_classes

        # step of optimization horizon
        self.step_horizon = 0

        # Init the energy system
        self.energy_system = EnergySystem(optimization_setup=self)

        # The time serier aggregation
        self.time_series_aggregation = None

        # set base scenario
        self.set_base_configuration()

        # add Elements to optimization
        self.add_elements()

    def add_elements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""
        logging.info("\n--- Add elements to model--- \n")

        for element_name in self.element_list:
            element_class = self.dict_element_classes[element_name]
            element_name = element_class.label
            element_set = self.system[element_name]

            # before adding the carriers, get set_carriers and check if carrier data exists
            if element_name == "set_carriers":
                element_set = self.energy_system.set_carriers
                self.system["set_carriers"] = element_set
                self.prepare.check_existing_carrier_data(self.system)

            # check if element_set has a subset and remove subset from element_set
            if element_name in self.analysis["subsets"].keys():
                element_subset = []
                for subset in self.analysis["subsets"][element_name]:
                    element_subset += [item for item in self.system[subset]]
                element_set = list(set(element_set) - set(element_subset))

            # add element class
            for item in element_set:
                self.add_element(element_class, item)
        if self.solver["analyze_numerics"]:
            self.energy_system.unit_handling.recommend_base_units(immutable_unit=self.solver["immutable_unit"],
                                                                  unit_exps=self.solver["rangeUnitExponents"])
        # conduct  time series aggregation
        self.time_series_aggregation = TimeSeriesAggregation(energy_system=self.energy_system)
        self.time_series_aggregation.conduct_tsa()

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
        """ get single element in class by name. Inherited by child classes.
        :param name: name of element
        :param cls: class of the elements to return
        :return element: return element whose name is matched """
        for _element in self.get_all_elements(cls=cls):
            if _element.name == name:
                return _element
        return None

    def get_attribute_of_all_elements(self, cls, attribute_name: str, capacity_types=False,
                                      return_attribute_is_series=False):
        """ get attribute values of all elements in a class
        :param cls: class of the elements to return
        :param attribute_name: str name of attribute
        :param capacity_types: boolean if attributes extracted for all capacity types
        :param return_attribute_is_series: boolean if information on attribute type is returned
        :return dict_of_attributes: returns dict of attribute values
        :return attribute_is_series: return information on attribute type """

        _class_elements = self.get_all_elements(cls=cls)
        dict_of_attributes = {}
        attribute_is_series = False
        for _element in _class_elements:
            if not capacity_types:
                dict_of_attributes, attribute_is_series = self.append_attribute_of_element_to_dict(_element, attribute_name, dict_of_attributes)
            # if extracted for both capacity types
            else:
                for capacity_type in self.system["set_capacity_types"]:
                    # append energy only for storage technologies
                    if capacity_type == self.system["set_capacity_types"][0] or _element.name in self.system["set_storage_technologies"]:
                        dict_of_attributes, attribute_is_series = self.append_attribute_of_element_to_dict(_element, attribute_name, dict_of_attributes, capacity_type)
        if return_attribute_is_series:
            return dict_of_attributes, attribute_is_series
        else:
            return dict_of_attributes

    def append_attribute_of_element_to_dict(self, _element, attribute_name, dict_of_attributes, capacity_type=None):
        """ get attribute values of all elements in this class
        :param _element: element of class
        :param attribute_name: str name of attribute
        :param dict_of_attributes: dict of attribute values
        :param capacity_type: capacity type for which attribute extracted. If None, not listed in key
        :return dict_of_attributes: returns dict of attribute values """

        attribute_is_series = False
        # add Energy for energy capacity type
        if capacity_type == self.system["set_capacity_types"][1]:
            attribute_name += "_energy"
        assert hasattr(_element, attribute_name), f"Element {_element.name} does not have attribute {attribute_name}"
        _attribute = getattr(_element, attribute_name)
        assert not isinstance(_attribute, pd.DataFrame), f"Not yet implemented for pd.DataFrames. Wrong format for element {_element.name}"
        # add attribute to dict_of_attributes
        if isinstance(_attribute, dict):
            dict_of_attributes.update({(_element.name,) + (key,): val for key, val in _attribute.items()})
        elif isinstance(_attribute, pd.Series) and "pwa" not in attribute_name:
            if capacity_type:
                _combined_key = (_element.name, capacity_type)
            else:
                _combined_key = _element.name
            if len(_attribute) > 1:
                dict_of_attributes[_combined_key] = _attribute
                attribute_is_series = True
            else:
                dict_of_attributes[_combined_key] = _attribute.squeeze()
                attribute_is_series = False
        elif isinstance(_attribute, int):
            if capacity_type:
                dict_of_attributes[(_element.name, capacity_type)] = [_attribute]
            else:
                dict_of_attributes[_element.name] = [_attribute]
        else:
            if capacity_type:
                dict_of_attributes[(_element.name, capacity_type)] = _attribute
            else:
                dict_of_attributes[_element.name] = _attribute
        return dict_of_attributes, attribute_is_series

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
        self.model = pe.ConcreteModel()
        # we need to reset the components to not carry them over
        self.variables = Variable()
        self.parameters = Parameter()
        self.constraints = Constraint()
        # define and construct components of self.model
        Element.construct_model_components(self.energy_system)
        logging.info("Apply Big-M GDP ")
        # add transformation factory so that disjuncts are solved
        pe.TransformationFactory("gdp.bigm").apply_to(self.model)
        # find smallest and largest coefficient and RHS
        self.analyze_numerics()

    def get_optimization_horizon(self):
        """ returns list of optimization horizon steps """
        # if using rolling horizon
        if self.system["use_rolling_horizon"]:
            self.years_in_horizon = self.system["years_in_rolling_horizon"]
            _time_steps_yearly = self.energy_system.set_time_steps_yearly
            self.steps_horizon = {year: list(range(year, min(year + self.years_in_horizon, max(_time_steps_yearly) + 1))) for year in _time_steps_yearly}
        # if no rolling horizon
        else:
            self.years_in_horizon = len(self.energy_system.set_time_steps_yearly)
            self.steps_horizon = {0: self.energy_system.set_time_steps_yearly}
        return list(self.steps_horizon.keys())

    def set_base_configuration(self, scenario="", elements={}):
        """set base configuration
        :param scenario: name of base scenario
        :param elements: elements in base configuration """
        self.base_scenario = scenario
        self.base_configuration = elements

    def restore_base_configuration(self, scenario, elements):
        """restore default configuration
        :param scenario: scenario name
        :param elements: dictionary of scenario dependent elements and parameters"""
        if not scenario == self.base_scenario:
            # restore base configuration
            self.overwrite_params(self.base_scenario, self.base_configuration)
            # continuously update base_configuration so all parameters are reset to their base value after being changed
            for element_name, params in elements.items():
                if element_name not in self.base_configuration.keys():
                    self.base_configuration[element_name] = params
                else:
                    for param in params:
                        if param not in self.base_configuration[element_name]:
                            self.base_configuration[element_name].append(param)

    def overwrite_params(self, scenario, elements):
        """overwrite scenario dependent parameters
        :param scenario: scenario name
        :param elements: dictionary of scenario dependent elements and parameters"""
        if scenario != "":
            scenario = "_" + scenario
        # list of parameters with raw_time_series
        conduct_tsa = False
        # overwrite scenario dependent parameter values for all elements
        for element_name, params in elements.items():
            if element_name == "EnergySystem":
                element = self.energy_system
            else:
                element = self.energy_system.get_element(Element, element_name)
            # overwrite scenario dependent parameters
            for param in params:
                assert "pwa" not in param, "Scenarios are not implemented for piece-wise affine parameters."
                file_name = param
                column = None
                if type(param) is tuple:
                    file_name, column = param
                    param = param[1]
                if "yearly_variation" in param:
                    param = param.replace("_yearly_variation", "")
                    file_name = param
                # get old param value
                _old_param = getattr(element, param)
                _index_names = _old_param.index.names
                _index_sets = [index_set for index_set, index_name in element.data_input.index_names.items() if index_name in _index_names]
                _time_steps = None
                # if existing capacity is changed, setExistingTechnologies, existing lifetime, and capexExistingCapacity have to be updated as well
                if "set_existing_technologies" in _index_sets:
                    # update setExistingTechnologies and existingLifetime
                    _existing_technologies = element.data_input.extract_set_existing_technologies(scenario=scenario)
                    _lifetime_existing_technologies = element.data_input.extract_lifetime_existing_technology(param, index_sets=_index_sets, scenario=scenario)
                    setattr(element, "set_existing_technologies", _existing_technologies)
                    setattr(element, "lifetime_existing_technology", _lifetime_existing_technologies)
                # set new parameter value
                if hasattr(element, "raw_time_series") and param in element.raw_time_series.keys():
                    conduct_tsa = True
                    _time_steps = self.energy_system.set_base_time_steps_yearly
                    element.raw_time_series[param] = element.data_input.extract_input_data(file_name, index_sets=_index_sets, time_steps=_time_steps, scenario=scenario)
                else:
                    assert isinstance(_old_param, pd.Series) or isinstance(_old_param, pd.DataFrame), f"Param values of '{param}' have to be a pd.DataFrame or pd.Series."
                    if "time" in _index_names:
                        # _time_steps = list(_old_param.index.unique("time"))
                        _time_steps = self.energy_system.set_base_time_steps_yearly
                    elif "year" in _index_names:
                        _time_steps = self.energy_system.set_time_steps_yearly
                    _new_param = element.data_input.extract_input_data(file_name, index_sets=_index_sets, time_steps=_time_steps, scenario=scenario)
                    setattr(element, param, _new_param)
                    # if existing capacity is changed, capexExistingCapacity also has to be updated
                    if "existing_capacity" in param:
                        storage_energy = False
                        if element in self.energy_system.system["set_storage_technologies"]:
                            storage_energy = True
                        _capex_existing_capacities = element.calculate_capex_of_existing_capacities(storage_energy=storage_energy)
                        setattr(element, "capex_existing_capacity", _capex_existing_capacities)
        # if scenario contains timeSeries dependent params conduct tsa
        if conduct_tsa:
            # we need to reset the Aggregation because the energy system might have changed
            self.time_series_aggregation = TimeSeriesAggregation(energy_system=self.energy_system)
            self.time_series_aggregation.conduct_tsa()

    def overwrite_time_indices(self, step_horizon):
        """ select subset of time indices, matching the step horizon
        :param step_horizon: step of the rolling horizon """

        if self.system["use_rolling_horizon"]:
            _time_steps_yearly_horizon = self.steps_horizon[step_horizon]
            _base_time_steps_horizon = self.energy_system.decode_yearly_time_steps(_time_steps_yearly_horizon)
            # overwrite time steps of each element
            for element in self.get_all_elements(Element):
                element.overwrite_time_steps(_base_time_steps_horizon)
            # overwrite base time steps and yearly base time steps
            _new_base_time_steps_horizon = _base_time_steps_horizon.squeeze().tolist()
            if not isinstance(_new_base_time_steps_horizon, list):
                _new_base_time_steps_horizon = [_new_base_time_steps_horizon]
            self.energy_system.set_base_time_steps = _new_base_time_steps_horizon
            self.energy_system.set_time_steps_yearly = _time_steps_yearly_horizon

    def analyze_numerics(self):
        """ get largest and smallest matrix coefficients and RHS """
        if self.solver["analyze_numerics"]:
            largest_rhs = [None, 0]
            smallest_rhs = [None, np.inf]
            largest_coeff = [None, 0]
            smallest_coeff = [None, np.inf]

            for cons in self.model.component_objects(pe.Constraint):
                for idx in cons:
                    deco_lhs = decompose_term(cons[idx].expr.args[0])
                    deco_rhs = decompose_term(cons[idx].expr.args[1])
                    deco_comb = []
                    if deco_lhs[0]:
                        deco_comb += deco_lhs[1]
                    if deco_rhs[0]:
                        deco_comb += deco_rhs[1]
                    _RHS = 0
                    for item in deco_comb:
                        _abs = abs(item[0])
                        if _abs != 0:
                            if item[1] is not None:
                                if _abs > largest_coeff[1]:
                                    largest_coeff[0] = (cons[idx].name, item[1].name)
                                    largest_coeff[1] = _abs
                                if _abs < smallest_coeff[1]:
                                    smallest_coeff[0] = (cons[idx].name, item[1].name)
                                    smallest_coeff[1] = _abs
                            else:
                                _RHS += item[0]
                    _RHS = abs(_RHS)
                    if _RHS != 0:
                        if _RHS > largest_rhs[1]:
                            largest_rhs[0] = cons[idx].name
                            largest_rhs[1] = _RHS
                        if _RHS < smallest_rhs[1]:
                            smallest_rhs[0] = cons[idx].name
                            smallest_rhs[1] = _RHS
            logging.info(
                f"Numeric Range Statistics:\nLargest Matrix Coefficient: {largest_coeff[1]} in {largest_coeff[0]}\nSmallest Matrix Coefficient: {smallest_coeff[1]} in {smallest_coeff[0]}\nLargest RHS: {largest_rhs[1]} in {largest_rhs[0]}\nSmallest RHS: {smallest_rhs[1]} in {smallest_rhs[0]}")

    def solve(self, solver):
        """Create model instance by assigning parameter values and instantiating the sets
        :param solver: dictionary containing the solver settings """

        solver_name = solver["name"]
        # remove options that are None
        solver_options = {key: solver["solver_options"][key] for key in solver["solver_options"] if solver["solver_options"][key] is not None}

        logging.info(f"\n--- Solve model instance using {solver_name} ---\n")
        # disable logger temporarily
        logging.disable(logging.WARNING)
        # write an ILP file to print the IIS if infeasible
        # (gives Warning: unable to write requested result file ".//outputs//logs//model.ilp" if feasible)
        solver_parameters = f"ResultFile={os.path.dirname(solver['solver_options']['logfile'])}//infeasibleModelIIS.ilp"

        if solver_name == "gurobi_persistent":
            self.opt = pe.SolverFactory(solver_name, options=solver_options)
            self.opt.set_instance(self.model, symbolic_solver_labels=solver["useSymbolicLabels"])
            self.results = self.opt.solve(tee=solver["verbosity"], logfile=solver["solver_options"]["logfile"], options_string=solver_parameters)
        else:
            self.opt = pe.SolverFactory(solver_name)
            self.results = self.opt.solve(self.model, tee=solver["verbosity"], keepfiles=True, logfile=solver["solver_options"]["logfile"])
        # enable logger
        logging.disable(logging.NOTSET)

        # store the solution into the results
        self.model.solutions.store_to(self.results, skip_stale_vars=True)

    def add_newly_built_capacity(self, step_horizon):
        """ adds the newly built capacity to the existing capacity
        :param step_horizon: step of the rolling horizon """
        if self.system["use_rolling_horizon"]:
            _built_capacity = pd.Series(self.model.built_capacity.extract_values())
            _invest_capacity = pd.Series(self.model.invested_capacity.extract_values())
            _capex = pd.Series(self.model.capex.extract_values())
            _rounding_value = 10 ** (-self.solver["rounding_decimal_points"])
            _built_capacity[_built_capacity <= _rounding_value] = 0
            _invest_capacity[_invest_capacity <= _rounding_value] = 0
            _capex[_capex <= _rounding_value] = 0
            _base_time_steps = self.energy_system.decode_yearly_time_steps([step_horizon])
            for tech in self.energy_system.get_all_elements(Technology):
                # new capacity
                _built_capacity_tech = _built_capacity.loc[tech.name].unstack()
                _invested_capacity_tech = _invest_capacity.loc[tech.name].unstack()
                _capex_tech = _capex.loc[tech.name].unstack()
                tech.add_newly_built_capacity_tech(_built_capacity_tech, _capex_tech, _base_time_steps)
                tech.add_newly_invested_capacity_tech(_invested_capacity_tech, step_horizon)

    def add_carbon_emission_cumulative(self, step_horizon):
        """ overwrite previous carbon emissions with cumulative carbon emissions
        :param step_horizon: step of the rolling horizon """
        if self.system["use_rolling_horizon"]:
            interval_between_years = self.energy_system.system["interval_between_years"]
            _carbon_emissions_cumulative = self.model.carbon_emissions_cumulative.extract_values()[step_horizon]
            carbon_emissions = self.model.carbon_emissions_total.extract_values()[step_horizon]
            self.energy_system.previous_carbon_emissions = _carbon_emissions_cumulative + carbon_emissions * (interval_between_years - 1)

    def initialize_component(self, calling_class, component_name, index_names=None, set_time_steps=None, capacity_types=False):
        """ this method initializes a modeling component by extracting the stored input data.
        :param calling_class: class from where the method is called
        :param component_name: name of modeling component
        :param index_names: names of index sets, only if calling_class is not EnergySystem
        :param set_time_steps: time steps, only if calling_class is EnergySystem
        :param capacity_types: boolean if extracted for capacities
        :return component_data: data to initialize the component """
        # if calling class is EnergySystem
        if calling_class == type(self):
            component = getattr(self, component_name)
            if index_names is not None:
                index_list = index_names
            elif set_time_steps is not None:
                index_list = [set_time_steps.name]
            else:
                index_list = []
            if set_time_steps:
                component_data = component[set_time_steps]
            elif type(component) == float:
                component_data = component
            else:
                component_data = component.squeeze()
        else:
            component_data, attribute_is_series = self.get_attribute_of_all_elements(calling_class, component_name, capacity_types=capacity_types, return_attribute_is_series=True)
            index_list = []
            if index_names:
                custom_set, index_list = calling_class.create_custom_set(index_names, self)
                if np.size(custom_set):
                    if attribute_is_series:
                        component_data = pd.concat(component_data, keys=component_data.keys())
                    else:
                        component_data = pd.Series(component_data)
                    component_data = self.check_for_subindex(component_data, custom_set)
            elif attribute_is_series:
                component_data = pd.concat(component_data, keys=component_data.keys())
            if not index_names:
                logging.warning(f"Initializing a parameter ({component_name}) without the specifying the index names will be deprecated!")

        return component_data, index_list

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

    def construct_objective(self):
        """ constructs the pe.Objective of the class <EnergySystem> """
        logging.info("Construct pe.Objective")

        # get selected objective rule
        if self.analysis["objective"] == "total_cost":
            objective_rule = self.rules.objective_total_cost_rule
        elif self.analysis["objective"] == "total_carbon_emissions":
            objective_rule = self.rules.objective_total_carbon_emissions_rule
        elif self.analysis["objective"] == "risk":
            logging.info("Objective of minimizing risk not yet implemented")
            objective_rule = self.rules.objective_risk_rule
        else:
            raise KeyError(f"Objective type {self.analysis['objective']} not known")

        # get selected objective sense
        if self.analysis["sense"] == "minimize":
            objective_sense = pe.minimize
        elif self.analysis["sense"] == "maximize":
            objective_sense = pe.maximize
        else:
            raise KeyError(f"Objective sense {self.analysis['sense']} not known")

        # construct objective
        self.pyomo_model.objective = pe.Objective(rule=objective_rule, sense=objective_sense)


class Objectives:
    """
    This class contains the objectives for the obtimization setup
    """
    def __init__(self, optimization_setup):
        """
        Inits the objectives
        :param optimization_setup: The optimization setup for the objectives
        """
        self.optimization_setup = optimization_setup

    # objective rules
    def objective_total_cost_rule(self, model):
        """objective function to minimize the total cost"""
        system = self.optimization_setup.system
        return (sum(model.NPV[year] * # discounted utility function
                    ((1 / (1 + system["social_discount_rate"])) ** (system["interval_between_years"] * (year - model.set_time_steps_yearly.at(1)))) for year in model.set_time_steps_yearly))

    def objectiveNPVRule(self, model):
        """ objective function to minimize NPV """
        return (sum(model.NPV[year] for year in model.set_time_steps_yearly))

    def objective_total_carbon_emissions_rule(self, model):
        """objective function to minimize total emissions"""
        return (sum(model.carbon_emissions_total[year] for year in model.set_time_steps_yearly))

    def objective_risk_rule(self, model):
        """objective function to minimize total risk"""
        # TODO implement objective functions for risk
        return pe.Constraint.Skip