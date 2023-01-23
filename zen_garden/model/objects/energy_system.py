"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        January-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard EnergySystem. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import copy
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import pyomo.environ as pe

from zen_garden.preprocess.functions.extract_input_data import DataInput
from zen_garden.preprocess.functions.unit_handling import UnitHandling
from .component import Parameter, Variable, Constraint
from .time_steps import SequenceTimeStepsDicts


class EnergySystem:

    # empty dict of element classes
    dict_element_classes = {}
    # empty list of class names
    element_list = {}

    def __init__(self, name_energy_system, analysis, system, paths, solver):
        """ initialization of the energy_system
        :param name_energy_system: name of energy_system that is added to the model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: paths to input folders of data
        :param solver: dictionary defining the solver"""

        # set attributes
        self.name = name_energy_system
        # set analysis
        self.analysis = analysis
        # set system
        self.system = system
        # set input paths
        self.paths = paths
        # set solver
        self.solver = solver
        # empty list of indexing sets
        self.indexing_sets = []
        # pe.ConcreteModel
        self.pyomo_model = None
        # empty dict of elements (will be filled with class_name: instance_list)
        self.dict_elements = defaultdict(list)
        # empty dict of technologies of carrier
        self.dict_technology_of_carrier = {}
        # empty dict of sequence of time steps operation
        self.dict_sequence_time_steps_operation = {}
        # empty dict of sequence of time steps yearly
        self.dict_sequence_time_steps_yearly = {}
        # empty dict of conversion from energy time steps to power time steps for storage technologies
        self.dict_time_steps_energy2power = {}
        # empty dict of conversion from operational time steps to invest time steps for technologies
        self.dict_time_steps_operation2invest = {}
        # empty dict of matching the last time step of the year in the storage domain to the first
        self.dict_time_steps_storage_level_startend_year = {}
        # The timesteps
        self.sequence_time_steps = SequenceTimeStepsDicts()

        # the components
        self.variables = None
        self.parameters = None
        self.constraints = None

        # set indexing sets
        for key in system:
            if "set" in key:
                self.indexing_sets.append(key)

        # set input path
        _folder_label = self.analysis["folder_name_system_specification"]
        self.input_path = self.paths[_folder_label]["folder"]

        # create UnitHandling object
        self.unit_handling = UnitHandling(self.input_path, self.solver["rounding_decimal_points"])

        # create DataInput object
        self.data_input = DataInput(element=self, system=self.system, analysis=self.analysis, solver=self.solver,
                                    energy_system=self, unit_handling=self.unit_handling)

        # store input data
        self.store_input_data()

        # create the rules
        self.rules = EnergySystemRules(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """

        # in class <EnergySystem>, all sets are constructed
        self.set_nodes = self.data_input.extract_locations()
        self.set_nodes_on_edges = self.calculate_edges_from_nodes()
        self.set_edges = list(self.set_nodes_on_edges.keys())
        self.set_carriers = []
        self.set_technologies = self.system["set_technologies"]
        # base time steps
        self.set_base_time_steps = list(range(0, self.system["unaggregated_time_steps_per_year"] * self.system["optimized_years"]))
        self.set_base_time_steps_yearly = list(range(0, self.system["unaggregated_time_steps_per_year"]))

        # yearly time steps
        self.set_time_steps_yearly = list(range(self.system["optimized_years"]))
        self.set_time_steps_yearly_entire_horizon = copy.deepcopy(self.set_time_steps_yearly)
        time_steps_yearly_duration = self.calculate_time_step_duration(self.set_time_steps_yearly)
        self.sequence_time_steps_yearly = np.concatenate([[time_step] * time_steps_yearly_duration[time_step] for time_step in time_steps_yearly_duration])
        self.set_sequence_time_steps(None, self.sequence_time_steps_yearly, time_step_type="yearly")
        # list containing simulated years (needed for convert_real_to_generic_time_indices() in extract_input_data.py)
        self.set_time_step_years = list(range(self.system["reference_year"],self.system["reference_year"] + self.system["optimized_years"]*self.system["interval_between_years"],self.system["interval_between_years"]))
        # parameters whose time-dependant data should not be interpolated (for years without data) in the extract_input_data.py convertRealToGenericTimeIndices() function
        self.parameters_interpolation_off = self.data_input.read_input_data("parameters_interpolation_off")
        # technology-specific
        self.set_conversion_technologies = self.system["set_conversion_technologies"]
        self.set_transport_technologies = self.system["set_transport_technologies"]
        self.set_storage_technologies = self.system["set_storage_technologies"]
        # carbon emissions limit
        self.carbon_emissions_limit = self.data_input.extract_input_data("carbon_emissions_limit", index_sets=["set_time_steps_yearly"], time_steps=self.set_time_steps_yearly)
        _fraction_year = self.system["unaggregated_time_steps_per_year"] / self.system["total_hours_per_year"]
        self.carbon_emissions_limit = self.carbon_emissions_limit * _fraction_year  # reduce to fraction of year
        self.carbon_emissions_budget = self.data_input.extract_input_data("carbon_emissions_budget", index_sets=[])
        self.previous_carbon_emissions = self.data_input.extract_input_data("previous_carbon_emissions", index_sets=[])
        # carbon price
        self.carbon_price = self.data_input.extract_input_data("carbon_price", index_sets=["set_time_steps_yearly"], time_steps=self.set_time_steps_yearly)
        self.carbon_price_overshoot = self.data_input.extract_input_data("carbon_price_overshoot", index_sets=[])

    def calculate_edges_from_nodes(self):
        """ calculates set_nodes_on_edges from set_nodes
        :return set_nodes_on_edges: dict with edges and corresponding nodes """

        set_nodes_on_edges = {}
        # read edge file
        set_edges_input = self.data_input.extract_locations(extract_nodes=False)
        if set_edges_input is not None:
            for edge in set_edges_input.index:
                set_nodes_on_edges[edge] = (set_edges_input.loc[edge, "node_from"], set_edges_input.loc[edge, "node_to"])
        else:
            logging.warning(f"DeprecationWarning: Implicit creation of edges will be deprecated. Provide 'set_edges.csv' in folder '{self.system['''folder_name_system_specification''']}' instead!")
            for node_from in self.set_nodes:
                for node_to in self.set_nodes:
                    if node_from != node_to:
                        set_nodes_on_edges[node_from + "-" + node_to] = (node_from, node_to)
        return set_nodes_on_edges

    def calculate_time_step_duration(self, input_time_steps, manual_base_time_steps=None):
        """ calculates (equidistant) time step durations for input time steps
        :param input_time_steps: input time steps
        :param manual_base_time_steps: manual list of base time steps
        :return time_step_duration_dict: dict with duration of each time step """
        if manual_base_time_steps is not None:
            base_time_steps = manual_base_time_steps
        else:
            base_time_steps = self.set_base_time_steps
        duration_input_time_steps = len(base_time_steps) / len(input_time_steps)
        time_step_duration_dict = {time_step: int(duration_input_time_steps) for time_step in input_time_steps}
        if not duration_input_time_steps.is_integer():
            logging.warning(f"The duration of each time step {duration_input_time_steps} of input time steps {input_time_steps} does not evaluate to an integer. \n"
                            f"The duration of the last time step is set to compensate for the difference")
            duration_last_time_step = len(base_time_steps) - sum(time_step_duration_dict[key] for key in time_step_duration_dict if key != input_time_steps[-1])
            time_step_duration_dict[input_time_steps[-1]] = duration_last_time_step
        return time_step_duration_dict

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

    def set_pyomo_model(self, pyomo_model):
        """ sets empty concrete model to energy_system
        :param pyomo_model: pe.ConcreteModel"""
        self.pyomo_model = pyomo_model
        # we need to reset the components to not carry them over
        self.variables = Variable()
        self.parameters = Parameter()
        self.constraints = Constraint()

    def set_manual_set_to_indexing_sets(self, manual_set):
        """ manually set to cls.indexing_sets """
        self.indexing_sets.append(manual_set)

    ### CLASS METHODS ###
    # setter/getter classmethods

    def set_technology_of_carrier(self, technology, list_technology_of_carrier):
        """ appends technology to carrier in dict_technology_of_carrier
        :param technology: name of technology in model
        :param list_technology_of_carrier: list of carriers correspondent to technology"""
        for carrier in list_technology_of_carrier:
            if carrier not in self.dict_technology_of_carrier:
                self.dict_technology_of_carrier[carrier] = [technology]
                self.set_carriers.append(carrier)
            elif technology not in self.dict_technology_of_carrier[carrier]:
                self.dict_technology_of_carrier[carrier].append(technology)

    def set_time_steps_energy2power(self, element, time_steps_energy2power):
        """ sets the dict of converting the energy time steps to the power time steps of storage technologies """
        self.dict_time_steps_energy2power[element] = time_steps_energy2power

    def set_time_steps_operation2invest(self, element, time_steps_operation2invest):
        """ sets the dict of converting the operational time steps to the invest time steps of all technologies """
        self.dict_time_steps_operation2invest[element] = time_steps_operation2invest

    def set_time_steps_storage_startend(self, element):
        """ sets the dict of matching the last time step of the year in the storage level domain to the first """
        _unaggregated_time_steps = self.system["unaggregated_time_steps_per_year"]
        _sequence_time_steps = self.get_sequence_time_steps(element + "_storage_level")
        _counter = 0
        _time_steps_start = []
        _time_steps_end = []
        while _counter < len(_sequence_time_steps):
            _time_steps_start.append(_sequence_time_steps[_counter])
            _counter += _unaggregated_time_steps
            _time_steps_end.append(_sequence_time_steps[_counter - 1])
        self.dict_time_steps_storage_level_startend_year[element] = {_start: _end for _start, _end in zip(_time_steps_start, _time_steps_end)}

    def set_sequence_time_steps(self, element, sequence_time_steps, time_step_type=None):
        """ sets sequence of time steps, either of operation, invest, or year
        :param element: name of element in model
        :param sequence_time_steps: list of time steps corresponding to base time step
        :param time_step_type: type of time step (operation or yearly)"""

        self.sequence_time_steps.set_sequence_time_steps(element=element, sequence_time_steps=sequence_time_steps, time_step_type=time_step_type)

    def set_sequence_time_steps_dict(self, dict_all_sequence_time_steps):
        """ sets all dicts of sequences of time steps.
        :param dict_all_sequence_time_steps: dict of all dict_sequence_time_steps"""
        self.sequence_time_steps.reset_dicts(dict_all_sequence_time_steps=dict_all_sequence_time_steps)

    def get_element_list(self):
        """ get attribute value of energy_system
        :param attribute_name: str name of attribute
        :return attribute: returns attribute values """
        element_classes = self.dict_element_classes.keys()
        carrier_classes = [element_name for element_name in element_classes if "Carrier" in element_name]
        technology_classes = [element_name for element_name in element_classes if "Technology" in element_name]
        self.element_list = technology_classes + carrier_classes
        return self.element_list

    def get_technology_of_carrier(self, carrier):
        """ gets technologies which are connected by carrier
        :param carrier: carrier which connects technologies
        :return listOfTechnologies: list of technologies connected by carrier"""
        if carrier in self.dict_technology_of_carrier:
            return self.dict_technology_of_carrier[carrier]
        else:
            return None

    def get_time_steps_energy2power(self, element):
        """ gets the dict of converting the energy time steps to the power time steps of storage technologies """
        return self.dict_time_steps_energy2power[element]

    def get_time_steps_operation2invest(self, element):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        return self.dict_time_steps_operation2invest[element]

    def get_time_steps_storage_startend(self, element, time_step):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        if time_step in self.dict_time_steps_storage_level_startend_year[element].keys():
            return self.dict_time_steps_storage_level_startend_year[element][time_step]
        else:
            return None

    def get_sequence_time_steps(self, element, time_step_type=None):
        """ get sequence ot time steps of element
        :param element: name of element in model
        :param time_step_type: type of time step (operation or invest)
        :return sequence_time_steps: list of time steps corresponding to base time step"""

        return self.sequence_time_steps.get_sequence_time_steps(element=element, time_step_type=time_step_type)

    def get_sequence_time_steps_dict(self):
        """ returns all dicts of sequence of time steps.
        :return dict_all_sequence_time_steps: dict of all dict_sequence_time_steps"""

        return self.sequence_time_steps.get_sequence_time_steps_dict()

    def calculate_connected_edges(self, node, direction: str):
        """ calculates connected edges going in (direction = 'in') or going out (direction = 'out')
        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode, out: node = startnode
        :return _set_connected_edges: list of connected edges """
        if direction == "in":
            # second entry is node into which the flow goes
            _set_connected_edges = [edge for edge in self.set_nodes_on_edges if self.set_nodes_on_edges[edge][1] == node]
        elif direction == "out":
            # first entry is node out of which the flow starts
            _set_connected_edges = [edge for edge in self.set_nodes_on_edges if self.set_nodes_on_edges[edge][0] == node]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return _set_connected_edges

    def calculate_reversed_edge(self, edge):
        """ calculates the reversed edge corresponding to an edge
        :param edge: input edge
        :return _reversed_edge: edge which corresponds to the reversed direction of edge"""
        _node_out, _node_in = self.set_nodes_on_edges[edge]
        for _reversed_edge in self.set_nodes_on_edges:
            if _node_out == self.set_nodes_on_edges[_reversed_edge][1] and _node_in == self.set_nodes_on_edges[_reversed_edge][0]:
                return _reversed_edge
        raise KeyError(f"Edge {edge} has no reversed edge. However, at least one transport technology is bidirectional")

    def decode_time_step(self, element, element_time_step: int, time_step_type: str = None):
        """ decodes time_step, i.e., retrieves the base_time_step corresponding to the variableTimeStep of a element.
        time_step of element --> base_time_step of model
        :param element: element of model, i.e., carrier or technology
        :param element_time_step: time step of element
        :param time_step_type: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model """

        return self.sequence_time_steps.decode_time_step(element=element, element_time_step=element_time_step, time_step_type=time_step_type)

    def encode_time_step(self, element: str, base_time_steps: int, time_step_type: str = None, yearly=False):
        """ encodes base_time_step, i.e., retrieves the time step of a element corresponding to base_time_step of model.
        base_time_step of model --> time_step of element
        :param element: name of element in model, i.e., carrier or technology
        :param base_time_steps: base time step of model for which the corresponding time index is extracted
        :param time_step_type: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element"""

        return self.sequence_time_steps.encode_time_step(element=element, base_time_steps=base_time_steps, time_step_type=time_step_type, yearly=yearly)

    def decode_yearly_time_steps(self, element_time_steps):
        """ decodes list of years to base time steps
        :param element_time_steps: time steps of year
        :return _full_base_time_steps: full list of time steps """
        _list_base_time_steps = []
        for year in element_time_steps:
            _list_base_time_steps.append(self.decode_time_step(None, year, "yearly"))
        _full_base_time_steps = np.concatenate(_list_base_time_steps)
        return _full_base_time_steps

    def convert_time_step_energy2power(self, element, timeStepEnergy):
        """ converts the time step of the energy quantities of a storage technology to the time step of the power quantities """
        _timeStepsEnergy2Power = self.get_time_steps_energy2power(element)
        return _timeStepsEnergy2Power[timeStepEnergy]

    def convert_time_step_operation2invest(self, element, time_step_operation):
        """ converts the operational time step to the invest time step """
        time_steps_operation2invest = self.get_time_steps_operation2invest(element)
        return time_steps_operation2invest[time_step_operation]

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

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to EnergySystem --- ###

    def construct_sets(self):
        """ constructs the pe.Sets of the class <EnergySystem> """
        # construct pe.Sets of the class <EnergySystem>

        # nodes
        self.pyomo_model.set_nodes = pe.Set(initialize=self.set_nodes, doc='Set of nodes')
        # edges
        self.pyomo_model.set_edges = pe.Set(initialize=self.set_edges, doc='Set of edges')
        # nodes on edges
        self.pyomo_model.set_nodes_on_edges = pe.Set(self.pyomo_model.set_edges, initialize=self.set_nodes_on_edges, doc='Set of nodes that constitute an edge. Edge connects first node with second node.')
        # carriers
        self.pyomo_model.set_carriers = pe.Set(initialize=self.set_carriers, doc='Set of carriers')
        # technologies
        self.pyomo_model.set_technologies = pe.Set(initialize=self.set_technologies, doc='Set of technologies')
        # all elements
        self.pyomo_model.set_elements = pe.Set(initialize=self.pyomo_model.set_technologies | self.pyomo_model.set_carriers, doc='Set of elements')
        # set set_elements to indexing_sets
        self.set_manual_set_to_indexing_sets("set_elements")
        # time-steps
        self.pyomo_model.set_base_time_steps = pe.Set(initialize=self.set_base_time_steps, doc='Set of base time-steps')
        # yearly time steps
        self.pyomo_model.set_time_steps_yearly = pe.Set(initialize=self.set_time_steps_yearly, doc='Set of yearly time-steps')
        # yearly time steps of entire optimization horizon
        self.pyomo_model.set_time_steps_yearly_entire_horizon = pe.Set(initialize=self.set_time_steps_yearly_entire_horizon, doc='Set of yearly time-steps of entire optimization horizon')

    def construct_params(self):
        """ constructs the pe.Params of the class <EnergySystem> """
        # carbon emissions limit
        cls = self.__class__
        self.parameters.add_parameter(name="carbon_emissions_limit", data=self.initialize_component(cls, "carbon_emissions_limit", set_time_steps=self.pyomo_model.set_time_steps_yearly),
            doc='Parameter which specifies the total limit on carbon emissions')
        # carbon emissions budget
        self.parameters.add_parameter(name="carbon_emissions_budget", data=self.initialize_component(cls, "carbon_emissions_budget"),
            doc='Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon')
        # carbon emissions budget
        self.parameters.add_parameter(name="previous_carbon_emissions", data=self.initialize_component(cls, "previous_carbon_emissions"), doc='Parameter which specifies the total previous carbon emissions')
        # carbon price
        self.parameters.add_parameter(name="carbon_price", data=self.initialize_component(cls, "carbon_price", set_time_steps=self.pyomo_model.set_time_steps_yearly),
            doc='Parameter which specifies the yearly carbon price')
        # carbon price of overshoot
        self.parameters.add_parameter(name="carbon_price_overshoot", data=self.initialize_component(cls, "carbon_price_overshoot"), doc='Parameter which specifies the carbon price for budget overshoot')

    def construct_vars(self):
        """ constructs the pe.Vars of the class <EnergySystem> """
        # carbon emissions
        self.variables.add_variable(self.pyomo_model, name="carbon_emissions_total", index_sets=self.pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total carbon emissions of energy system")
        # cumulative carbon emissions
        self.variables.add_variable(self.pyomo_model, name="carbon_emissions_cumulative", index_sets=self.pyomo_model.set_time_steps_yearly, domain=pe.Reals,
            doc="cumulative carbon emissions of energy system over time for each year")
        # carbon emission overshoot
        self.variables.add_variable(self.pyomo_model, name="carbon_emissions_overshoot", index_sets=self.pyomo_model.set_time_steps_yearly, domain=pe.NonNegativeReals,
            doc="overshoot carbon emissions of energy system at the end of the time horizon")
        # cost of carbon emissions
        self.variables.add_variable(self.pyomo_model, name="cost_carbon_emissions_total", index_sets=self.pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total cost of carbon emissions of energy system")
        # costs
        self.variables.add_variable(self.pyomo_model, name="cost_total", index_sets=self.pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total cost of energy system")
        # NPV
        self.variables.add_variable(self.pyomo_model, name="NPV", index_sets=self.pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="NPV of energy system")

    def construct_constraints(self):
        """ constructs the pe.Constraints of the class <EnergySystem> """

        # carbon emissions
        self.constraints.add_constraint(self.pyomo_model, name="constraint_carbon_emissions_total", index_sets=self.pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_total_rule,
            doc="total carbon emissions of energy system")
        # carbon emissions
        self.constraints.add_constraint(self.pyomo_model, name="constraint_carbon_emissions_cumulative", index_sets=self.pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_cumulative_rule,
            doc="cumulative carbon emissions of energy system over time")
        # cost of carbon emissions
        self.constraints.add_constraint(self.pyomo_model, name="constraint_carbon_cost_total", index_sets=self.pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_cost_total_rule, doc="total carbon cost of energy system")
        # carbon emissions
        self.constraints.add_constraint(self.pyomo_model, name="constraint_carbon_emissions_limit", index_sets=self.pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_limit_rule,
            doc="limit of total carbon emissions of energy system")
        # carbon emissions
        self.constraints.add_constraint(self.pyomo_model, name="constraint_carbon_emissions_budget", index_sets=self.pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_budget_rule,
            doc="Budget of total carbon emissions of energy system")
        # costs
        self.constraints.add_constraint(self.pyomo_model, name="constraint_cost_total", index_sets=self.pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_cost_total_rule, doc="total cost of energy system")
        # NPV
        self.constraints.add_constraint(self.pyomo_model, name="constraint_NPV", index_sets=self.pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_NPV_rule, doc="NPV of energy system")

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


class EnergySystemRules:
    """
    This class takes care of the rules for the EnergySystem
    """

    def __init__(self, energy_system: EnergySystem):
        """
        Inits the constraints for a given energy syste,
        :param energy_system: The energy system used to build the constraints
        """

        self.energy_system = energy_system

    def constraint_carbon_emissions_total_rule(self, model, year):
        """ add up all carbon emissions from technologies and carriers """
        return (model.carbon_emissions_total[year] ==
                # technologies
                model.carbon_emissions_technology_total[year] + # carriers
                model.carbon_emissions_carrier_total[year])

    def constraint_carbon_emissions_cumulative_rule(self, model, year):
        """ cumulative carbon emissions over time """
        # get parameter object
        params = self.energy_system.parameters
        interval_between_years = self.energy_system.system["interval_between_years"]
        if year == model.set_time_steps_yearly.at(1):
            return (model.carbon_emissions_cumulative[year] == model.carbon_emissions_total[year] + params.previous_carbon_emissions)
        else:
            return (model.carbon_emissions_cumulative[year] == model.carbon_emissions_cumulative[year - 1] + model.carbon_emissions_total[year - 1] * (interval_between_years - 1) +
                    model.carbon_emissions_total[year])

    def constraint_carbon_cost_total_rule(self, model, year):
        """ carbon cost associated with the carbon emissions of the system in each year """
        # get parameter object
        params = self.energy_system.parameters
        return (model.cost_carbon_emissions_total[year] == params.carbon_price[year] * model.carbon_emissions_total[year] # add overshoot price
                + model.carbon_emissions_overshoot[year] * params.carbon_price_overshoot)

    def constraint_carbon_emissions_limit_rule(self, model, year):
        """ time dependent carbon emissions limit from technologies and carriers"""
        # get parameter object
        params = self.energy_system.parameters
        if params.carbon_emissions_limit[year] != np.inf:
            return (params.carbon_emissions_limit[year] >= model.carbon_emissions_total[year])
        else:
            return pe.Constraint.Skip

    def constraint_carbon_emissions_budget_rule(self, model, year):
        """ carbon emissions budget of entire time horizon from technologies and carriers.
        The prediction extends until the end of the horizon, i.e.,
        last optimization time step plus the current carbon emissions until the end of the horizon """
        # get parameter object
        params = self.energy_system.parameters
        interval_between_years = self.energy_system.system["interval_between_years"]
        if params.carbon_emissions_budget != np.inf:  # TODO check for last year - without last term?
            return (params.carbon_emissions_budget + model.carbon_emissions_overshoot[year] >= model.carbon_emissions_cumulative[year] + model.carbon_emissions_total[year] * (interval_between_years - 1))
        else:
            return pe.Constraint.Skip

    def constraint_cost_total_rule(self, model, year):
        """ add up all costs from technologies and carriers"""
        return (model.cost_total[year] ==
                # capex
                model.capex_total[year] + # opex
                model.opex_total[year] + # carrier costs
                model.cost_carrier_total[year] + # carbon costs
                model.cost_carbon_emissions_total[year])

    def constraint_NPV_rule(self, model, year):
        """ discounts the annual capital flows to calculate the NPV """
        system = self.energy_system.system
        discount_rate = self.energy_system.analysis["discount_rate"]
        if system["optimized_years"] > 1:
            interval_between_years = system["interval_between_years"]
        else:
            interval_between_years = 1

        return (model.NPV[year] == model.cost_total[year] * sum(# economic discount
            ((1 / (1 + discount_rate)) ** (interval_between_years * (year - model.set_time_steps_yearly.at(1)) + _intermediate_time_step)) for _intermediate_time_step in range(0, interval_between_years)))

    # objective rules
    def objective_total_cost_rule(self, model):
        """objective function to minimize the total cost"""
        system = self.energy_system.system
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
