"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        January-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard EnergySystem. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import logging
import warnings
import pyomo.environ as pe
import numpy         as np
import pandas        as pd
import copy
from zen_garden.preprocess.functions.extract_input_data import DataInput
from zen_garden.preprocess.functions.unit_handling      import UnitHandling
from .component import Parameter,Variable,Constraint
from .time_steps import SequenceTimeStepsDicts

class EnergySystem:
    # energy_system
    energy_system = None
    # pe.ConcreteModel
    pyomo_model = None
    # analysis
    analysis = None
    # system
    system = None
    # paths
    paths = None
    # solver
    solver = None
    # unit handling instance
    unit_handling = None
    # empty list of indexing sets
    indexing_sets = []
    # empty dict of technologies of carrier
    dict_technology_of_carrier = {}
    # empty dict of sequence of time steps operation
    dict_sequence_time_steps_operation = {}
    # empty dict of sequence of time steps yearly
    dict_sequence_time_steps_yearly = {}
    # empty dict of conversion from energy time steps to power time steps for storage technologies
    dict_time_steps_energy2power = {}
    # empty dict of conversion from operational time steps to invest time steps for technologies
    dict_time_steps_operation2invest = {}
    # empty dict of matching the last time step of the year in the storage domain to the first
    dict_time_steps_storage_level_startend_year = {}
    # empty dict of element classes
    dict_element_classes = {}
    # empty list of class names
    element_list = {}
    # The timesteps
    SequenceTimeSteps = SequenceTimeStepsDicts()

    def __init__(self,name_energy_system):
        """ initialization of the energy_system
        :param name_energy_system: name of energy_system that is added to the model """

        # only one energy system can be defined
        assert not EnergySystem.get_energy_system(), "Only one energy system can be defined."

        # set attributes
        self.name = name_energy_system

        # add energy_system to list
        EnergySystem.set_energy_system(self)

        # get input path
        self.get_input_path()

        # create UnitHandling object
        EnergySystem.create_unit_handling()

        # create DataInput object
        self.datainput = DataInput(self,EnergySystem.get_system(), EnergySystem.get_analysis(), EnergySystem.get_solver(), EnergySystem.get_energy_system(), self.unit_handling)

        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """

        system                          = EnergySystem.get_system()
        self.paths                      = EnergySystem.get_paths()

        # in class <EnergySystem>, all sets are constructed
        self.set_nodes              = self.datainput.extract_locations()
        self.set_nodes_on_edges        = self.calculate_edges_from_nodes()
        self.set_edges               = list(self.set_nodes_on_edges.keys())
        self.set_carriers            = []
        self.set_technologies        = system["set_technologies"]
        # base time steps
        self.set_base_time_steps       = list(range(0, system["unaggregated_time_steps_per_year"] * system["optimized_years"]))
        self.set_base_time_steps_yearly = list(range(0, system["unaggregated_time_steps_per_year"]))

        # yearly time steps
        self.set_time_steps_yearly = list(range(self.system["optimized_years"]))
        self.set_time_steps_yearly_entire_horizon = copy.deepcopy(self.set_time_steps_yearly)
        time_steps_yearly_duration = EnergySystem.calculate_time_step_duration(self.set_time_steps_yearly)
        self.sequence_time_steps_yearly = np.concatenate([[time_step] * time_steps_yearly_duration[time_step] for time_step in time_steps_yearly_duration])
        self.set_sequence_time_steps(None, self.sequence_time_steps_yearly, time_step_type="yearly")

        # technology-specific
        self.set_conversion_technologies = system["set_conversion_technologies"]
        self.set_transport_technologies = system["set_transport_technologies"]
        self.set_storage_technologies = system["set_storage_technologies"]
        # carbon emissions limit
        self.carbon_emissions_limit = self.datainput.extract_input_data("carbon_emissions_limit", index_sets=["set_time_steps"],
                                                                    time_steps=self.set_time_steps_yearly)
        _fraction_year = system["unaggregated_time_steps_per_year"] / system["total_hours_per_year"]
        self.carbon_emissions_limit = self.carbon_emissions_limit * _fraction_year  # reduce to fraction of year
        self.carbon_emissions_budget = self.datainput.extract_input_data("carbon_emissions_budget", index_sets=[])
        self.previous_carbon_emissions = self.datainput.extract_input_data("previous_carbon_emissions", index_sets=[])
        # carbon price
        self.carbon_price = self.datainput.extract_input_data("carbon_price", index_sets=["set_time_steps"],
                                                           time_steps=self.set_time_steps_yearly)
        self.carbon_price_overshoot = self.datainput.extract_input_data("carbon_price_overshoot", index_sets=[])

    def calculate_edges_from_nodes(self):
        """ calculates set_nodes_on_edges from set_nodes
        :return set_nodes_on_edges: dict with edges and corresponding nodes """
        system = EnergySystem.get_system()
        set_nodes_on_edges = {}
        # read edge file
        set_edges_input = self.datainput.extract_locations(extract_nodes=False)
        if set_edges_input is not None:
            for edge in set_edges_input.index:
                set_nodes_on_edges[edge] = (set_edges_input.loc[edge,"node_from"],set_edges_input.loc[edge,"node_to"])
        else:
            warnings.warn(f"Implicit creation of edges will be deprecated. Provide 'set_edges.csv' in folder '{system['''folder_name_system_specification''']}' instead!",FutureWarning)
            for node_from in self.set_nodes:
                for node_to in self.set_nodes:
                    if node_from != node_to:
                        set_nodes_on_edges[node_from+"-"+node_to] = (node_from,node_to)
        return set_nodes_on_edges

    def get_input_path(self):
        """ get input path where input data is stored input_path"""
        _folder_label = EnergySystem.get_analysis()["folder_name_system_specification"]

        paths = EnergySystem.get_paths()
        # get input path of energy system specification
        self.input_path = paths[_folder_label]["folder"]

    ### CLASS METHODS ###
    # setter/getter classmethods
    @classmethod
    def set_energy_system(cls,energy_system):
        """ set energy_system.
        :param energy_system: new energy_system that is set """
        cls.energy_system = energy_system

    @classmethod
    def set_optimization_attributes(cls,analysis, system,paths,solver):
        """ set attributes of class <EnergySystem> with inputs 
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: paths to input folders of data
        :param solver: dictionary defining the solver"""
        # set analysis
        cls.analysis = analysis
        # set system
        cls.system = system
        # set input paths
        cls.paths = paths
        # set solver
        cls.solver = solver
        # set indexing sets
        cls.set_indexing_sets()

    @classmethod
    def set_pyomo_model(cls,pyomo_model):
        """ sets empty concrete model to energy_system
        :param pyomo_model: pe.ConcreteModel"""
        cls.pyomo_model = pyomo_model

    @classmethod
    def set_indexing_sets(cls):
        """ set sets that serve as an index for other sets """
        system = cls.get_system()
        # iterate over sets
        for key in system:
            if "set" in key:
                cls.indexing_sets.append(key)

    @classmethod
    def set_manual_set_to_indexing_sets(cls,manual_set):
        """ manually set to cls.indexing_sets """
        cls.indexing_sets.append(manual_set)

    @classmethod
    def set_technology_of_carrier(cls,technology,list_technology_of_carrier):
        """ appends technology to carrier in dict_technology_of_carrier
        :param technology: name of technology in model
        :param list_technology_of_carrier: list of carriers correspondent to technology"""
        for carrier in list_technology_of_carrier:
            if carrier not in cls.dict_technology_of_carrier:
                cls.dict_technology_of_carrier[carrier] = [technology]
                cls.energy_system.set_carriers.append(carrier)
            elif technology not in cls.dict_technology_of_carrier[carrier]:
                cls.dict_technology_of_carrier[carrier].append(technology)

    @classmethod
    def set_time_steps_energy2power(cls, element, time_steps_energy2power):
        """ sets the dict of converting the energy time steps to the power time steps of storage technologies """
        cls.dict_time_steps_energy2power[element] = time_steps_energy2power

    @classmethod
    def set_time_steps_operation2invest(cls, element, time_steps_operation2invest):
        """ sets the dict of converting the operational time steps to the invest time steps of all technologies """
        cls.dict_time_steps_operation2invest[element] = time_steps_operation2invest

    @classmethod
    def set_time_steps_storage_startend(cls, element):
        """ sets the dict of matching the last time step of the year in the storage level domain to the first """
        system = cls.get_system()
        _unaggregated_time_steps  = system["unaggregated_time_steps_per_year"]
        _sequence_time_steps      = cls.get_sequence_time_steps(element + "_storage_level")
        _counter = 0
        _time_steps_start = []
        _time_steps_end = []
        while _counter < len(_sequence_time_steps):
            _time_steps_start.append(_sequence_time_steps[_counter])
            _counter += _unaggregated_time_steps
            _time_steps_end.append(_sequence_time_steps[_counter - 1])
        cls.dict_time_steps_storage_level_startend_year[element] = {_start: _end for _start, _end in zip(_time_steps_start, _time_steps_end)}

    @classmethod
    def set_sequence_time_steps(cls,element,sequence_time_steps,time_step_type = None):
        """ sets sequence of time steps, either of operation, invest, or year
        :param element: name of element in model
        :param sequenceTimeSteps: list of time steps corresponding to base time step
        :param timeStepType: type of time step (operation or yearly)"""

        cls.SequenceTimeSteps.setSequenceTimeSteps(element=element, sequenceTimeSteps=sequenceTimeSteps,
                                                   timeStepType=timeStepType)

    @classmethod
    def set_sequence_time_steps_dict(cls,dict_all_sequence_time_steps):
        """ sets all dicts of sequences of time steps.
        :param dictAllSequenceTimeSteps: dict of all dictSequenceTimeSteps"""
        cls.SequenceTimeSteps.reset_dicts(dictAllSequenceTimeSteps=dictAllSequenceTimeSteps)

    @classmethod
    def get_pyomo_model(cls):
        """ get pyomo_model of the class <EnergySystem>. Every child class can access model and add components.
        :return pyomo_model: pe.ConcreteModel """
        return cls.pyomo_model

    @classmethod
    def get_analysis(cls):
        """ get analysis of the class <EnergySystem>.
        :return analysis: dictionary defining the analysis framework """
        return cls.analysis

    @classmethod
    def get_system(cls):
        """ get system
        :return system: dictionary defining the system """
        return cls.system

    @classmethod
    def get_paths(cls):
        """ get paths
        :return paths: paths to folders of input data """
        return cls.paths

    @classmethod
    def get_solver(cls):
        """ get solver
        :return solver: dictionary defining the analysis solver """
        return cls.solver

    @classmethod
    def get_energy_system(cls):
        """ get energy_system.
        :return energy_system: return energy_system """
        return cls.energy_system

    @classmethod
    def get_element_list(cls):
        """ get attribute value of energy_system
        :param attribute_name: str name of attribute
        :return attribute: returns attribute values """
        element_classes    = cls.dict_element_classes.keys()
        carrier_classes    = [element_name for element_name in element_classes if "Carrier" in element_name]
        technology_classes = [element_name for element_name in element_classes if "Technology" in element_name]
        cls.element_list   = technology_classes + carrier_classes
        return cls.element_list

    @classmethod
    def get_attribute(cls,attribute_name:str):
        """ get attribute value of energy_system
        :param attribute_name: str name of attribute
        :return attribute: returns attribute values """
        energy_system = cls.get_energy_system()
        assert hasattr(energy_system,attribute_name), f"The energy system does not have attribute '{attribute_name}"
        return getattr(energy_system,attribute_name)

    @classmethod
    def get_indexing_sets(cls):
        """ set sets that serve as an index for other sets
        :return cls.indexing_sets: list of sets that serve as an index for other sets"""
        return cls.indexing_sets

    @classmethod
    def get_technology_of_carrier(cls,carrier):
        """ gets technologies which are connected by carrier
        :param carrier: carrier which connects technologies
        :return listOfTechnologies: list of technologies connected by carrier"""
        if carrier in cls.dict_technology_of_carrier:
            return cls.dict_technology_of_carrier[carrier]
        else:
            return None

    @classmethod
    def get_time_steps_energy2power(cls, element):
        """ gets the dict of converting the energy time steps to the power time steps of storage technologies """
        return cls.dict_time_steps_energy2power[element]

    @classmethod
    def get_time_steps_operation2invest(cls, element):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        return cls.dict_time_steps_operation2invest[element]

    @classmethod
    def get_time_steps_storage_startend(cls, element, time_step):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        if time_step in cls.dict_time_steps_storage_level_startend_year[element].keys():
            return cls.dict_time_steps_storage_level_startend_year[element][time_step]
        else:
            return None

    @classmethod
    def get_sequence_time_steps(cls,element,time_step_type = None):
        """ get sequence ot time steps of element
        :param element: name of element in model
        :param timeStepType: type of time step (operation or invest)
        :return sequenceTimeSteps: list of time steps corresponding to base time step"""

        return cls.SequenceTimeSteps.getSequenceTimeSteps(element=element, timeStepType=timeStepType)

    @classmethod
    def get_sequence_time_steps_dict(cls):
        """ returns all dicts of sequence of time steps.
        :return dictAllSequenceTimeSteps: dict of all dictSequenceTimeSteps"""

        return cls.SequenceTimeSteps.getSequenceTimeStepsDict()

    @classmethod
    def get_unit_handling(cls):
        """ returns the unit handling object """
        return cls.unit_handling

    @classmethod
    def create_unit_handling(cls):
        """ creates and stores the unit handling object """
        # create UnitHandling object
        cls.unit_handling = UnitHandling(cls.get_energy_system().input_path,cls.get_energy_system().solver["rounding_decimal_points"])

    @classmethod
    def calculate_connected_edges(cls,node,direction:str):
        """ calculates connected edges going in (direction = 'in') or going out (direction = 'out')
        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode, out: node = startnode
        :return _set_connected_edges: list of connected edges """
        energy_system = cls.get_energy_system()
        if direction == "in":
            # second entry is node into which the flow goes
            _set_connected_edges = [edge for edge in energy_system.set_nodes_on_edges if energy_system.set_nodes_on_edges[edge][1]==node]
        elif direction == "out":
            # first entry is node out of which the flow starts
            _set_connected_edges = [edge for edge in energy_system.set_nodes_on_edges if energy_system.set_nodes_on_edges[edge][0]==node]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return _set_connected_edges

    @classmethod
    def calculate_reversed_edge(cls, edge):
        """ calculates the reversed edge corresponding to an edge
        :param edge: input edge
        :return _reversed_edge: edge which corresponds to the reversed direction of edge"""
        energy_system = cls.get_energy_system()
        _node_out, _node_in = energy_system.set_nodes_on_edges[edge]
        for _reversed_edge in energy_system.set_nodes_on_edges:
            if _node_out == energy_system.set_nodes_on_edges[_reversed_edge][1] and _node_in == energy_system.set_nodes_on_edges[_reversed_edge][0]:
                return _reversed_edge
        raise KeyError(f"Edge {edge} has no reversed edge. However, at least one transport technology is bidirectional")

    @classmethod
    def calculate_time_step_duration(cls,input_time_steps,manual_base_time_steps = None):
        """ calculates (equidistant) time step durations for input time steps
        :param input_time_steps: input time steps
        :param manual_base_time_steps: manual list of base time steps
        :return time_step_duration_dict: dict with duration of each time step """
        if manual_base_time_steps is not None:
            base_time_steps       = manual_base_time_steps
        else:
            base_time_steps       = cls.get_energy_system().set_base_time_steps
        duration_input_time_steps  = len(base_time_steps)/len(input_time_steps)
        time_step_duration_dict    = {time_step: int(duration_input_time_steps) for time_step in input_time_steps}
        if not duration_input_time_steps.is_integer():
            logging.warning(f"The duration of each time step {duration_input_time_steps} of input time steps {input_time_steps} does not evaluate to an integer. \n"
                            f"The duration of the last time step is set to compensate for the difference")
            duration_last_time_step = len(base_time_steps) - sum(time_step_duration_dict[key] for key in time_step_duration_dict if key != input_time_steps[-1])
            time_step_duration_dict[input_time_steps[-1]] = duration_last_time_step
        return time_step_duration_dict

    @classmethod
    def decode_time_step(cls,element,element_time_step:int,time_step_type:str = None):
        """ decodes time_step, i.e., retrieves the base_time_step corresponding to the variableTimeStep of a element.
        time_step of element --> base_time_step of model
        :param element: element of model, i.e., carrier or technology
        :param elementTimeStep: time step of element
        :param timeStepType: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model """

        return cls.SequenceTimeSteps.decodeTimeStep(element=element, elementTimeStep=elementTimeStep,
                                                    timeStepType=timeStepType)

    @classmethod
    def encode_time_step(cls,element:str,base_time_steps:int,time_step_type:str = None,yearly=False):
        """ encodes base_time_step, i.e., retrieves the time step of a element corresponding to base_time_step of model.
        base_time_step of model --> time_step of element
        :param element: name of element in model, i.e., carrier or technology
        :param base_time_steps: base time step of model for which the corresponding time index is extracted
        :param time_step_type: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element"""

        return cls.SequenceTimeSteps.encodeTimeStep(element=element, baseTimeSteps=baseTimeSteps,
                                                    timeStepType=timeStepType, yearly=yearly)

    @classmethod
    def decode_yearly_time_steps(cls,element_time_steps):
        """ decodes list of years to base time steps
        :param element_time_steps: time steps of year
        :return _full_base_time_steps: full list of time steps """
        _list_base_time_steps = []
        for year in element_time_steps:
            _list_base_time_steps.append(cls.decode_time_step(None,year,"yearly"))
        _full_base_time_steps = np.concatenate(_list_base_time_steps)
        return _full_base_time_steps

    @classmethod
    def convert_time_step_energy2power(cls,element,timeStepEnergy):
        """ converts the time step of the energy quantities of a storage technology to the time step of the power quantities """
        _timeStepsEnergy2Power = cls.get_time_steps_energy2power(element)
        return _timeStepsEnergy2Power[timeStepEnergy]

    @classmethod
    def convert_time_step_operation2invest(cls, element, time_step_operation):
        """ converts the operational time step to the invest time step """
        time_steps_operation2invest = cls.get_time_steps_operation2invest(element)
        return time_steps_operation2invest[time_step_operation]

    @classmethod
    def initialize_component(cls,calling_class,component_name,index_names = None,set_time_steps = None,capacity_types = False):
        """ this method initializes a modeling component by extracting the stored input data.
        :param calling_class: class from where the method is called
        :param component_name: name of modeling component
        :param index_names: names of index sets, only if calling_class is not EnergySystem
        :param set_time_steps: time steps, only if calling_class is EnergySystem
        :param capacity_types: boolean if extracted for capacities
        :return component_data: data to initialize the component """
        # if calling class is EnergySystem
        if calling_class == cls:
            component = getattr(cls.get_energy_system(), component_name)
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
            component_data,attribute_is_series = calling_class.get_attribute_of_all_elements(component_name, capacity_types= capacity_types, return_attribute_is_series=True)
            index_list = []
            if index_names:
                custom_set,index_list = calling_class.create_custom_set(index_names)
                if np.size(custom_set):
                    if attribute_is_series:
                        component_data = pd.concat(component_data,keys=component_data.keys())
                    else:
                        component_data = pd.Series(component_data)
                    component_data   = cls.check_for_subindex(component_data,custom_set)
            elif attribute_is_series:
                component_data = pd.concat(component_data, keys=component_data.keys())
            if not index_names:
                logging.warning(f"Initializing a parameter ({component_name}) without the specifying the index names will be deprecated!")

        return component_data,index_list

    @classmethod
    def check_for_subindex(cls,component_data,custom_set):
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
            for _level,_shape in enumerate(_custom_index.levshape):
                if _shape == 1:
                    _reduced_custom_index = _reduced_custom_index.droplevel(_level)
            try:
                component_data = component_data[_reduced_custom_index]
                component_data.index     = _custom_index
                return component_data
            except KeyError:
                raise KeyError(f"the custom set {custom_set} cannot be used as a subindex of {component_data.index}")

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to EnergySystem --- ###
    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <EnergySystem> """
        # construct pe.Sets of the class <EnergySystem>
        model = cls.get_pyomo_model()
        energy_system = cls.get_energy_system()

        # nodes
        model.set_nodes = pe.Set(
            initialize=energy_system.set_nodes,
            doc='Set of nodes')
        # edges
        model.set_edges = pe.Set(
            initialize = energy_system.set_edges,
            doc = 'Set of edges')
        # nodes on edges
        model.set_nodes_on_edges = pe.Set(
            model.set_edges,
            initialize = energy_system.set_nodes_on_edges,
            doc = 'Set of nodes that constitute an edge. Edge connects first node with second node.')
        # carriers
        model.set_carriers = pe.Set(
            initialize=energy_system.set_carriers,
            doc='Set of carriers')
        # technologies
        model.set_technologies = pe.Set(
            initialize=energy_system.set_technologies,
            doc='Set of technologies')
        # all elements
        model.set_elements = pe.Set(
            initialize=model.set_technologies | model.set_carriers,
            doc='Set of elements')
        # set set_elements to indexing_sets
        cls.set_manual_set_to_indexing_sets("set_elements")
        # time-steps
        model.set_base_time_steps = pe.Set(
            initialize=energy_system.set_base_time_steps,
            doc='Set of base time-steps')
        # yearly time steps
        model.set_time_steps_yearly = pe.Set(
            initialize=energy_system.set_time_steps_yearly,
            doc='Set of yearly time-steps')
        # yearly time steps of entire optimization horizon
        model.set_time_steps_yearly_entire_horizon = pe.Set(
            initialize=energy_system.set_time_steps_yearly_entire_horizon,
            doc='Set of yearly time-steps of entire optimization horizon')

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <EnergySystem> """
        # get model
        model = cls.get_pyomo_model()

        # carbon emissions limit
        Parameter.add_parameter(
            name="carbon_emissions_limit",
            data=cls.initialize_component(cls, "carbon_emissions_limit", set_time_steps=model.set_time_steps_yearly),
            doc='Parameter which specifies the total limit on carbon emissions'
        )
        # carbon emissions budget
        Parameter.add_parameter(
            name="carbon_emissions_budget",
            data=cls.initialize_component(cls, "carbon_emissions_budget"),
            doc='Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon'
        )
        # carbon emissions budget
        Parameter.add_parameter(
            name="previous_carbon_emissions",
            data=cls.initialize_component(cls, "previous_carbon_emissions"),
            doc='Parameter which specifies the total previous carbon emissions'
        )
        # carbon price
        Parameter.add_parameter(
            name="carbon_price",
            data=cls.initialize_component(cls, "carbon_price", set_time_steps=model.set_time_steps_yearly),
            doc='Parameter which specifies the yearly carbon price'
        )
        # carbon price of overshoot
        Parameter.add_parameter(
            name="carbon_price_overshoot",
            data=cls.initialize_component(cls, "carbon_price_overshoot"),
            doc='Parameter which specifies the carbon price for budget overshoot'
        )

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <EnergySystem> """
        # get model
        model = cls.get_pyomo_model()

        # carbon emissions
        Variable.add_variable(
            model,
            name = "carbon_emissions_total",
            index_sets = model.set_time_steps_yearly,
            domain = pe.Reals,
            doc = "total carbon emissions of energy system"
        )
        # cumulative carbon emissions
        Variable.add_variable(
            model,
            name="carbon_emissions_cumulative",
            index_sets=model.set_time_steps_yearly,
            domain=pe.Reals,
            doc="cumulative carbon emissions of energy system over time for each year"
        )
        # carbon emission overshoot
        Variable.add_variable(
            model,
            name="carbon_emissions_overshoot",
            index_sets=model.set_time_steps_yearly,
            domain=pe.NonNegativeReals,
            doc="overshoot carbon emissions of energy system at the end of the time horizon"
        )
        # cost of carbon emissions
        Variable.add_variable(
            model,
            name="cost_carbon_emissions_total",
            index_sets = model.set_time_steps_yearly,
            domain=pe.Reals,
            doc="total cost of carbon emissions of energy system"
        )
        # costs
        Variable.add_variable(
            model,
            name="cost_total",
            index_sets = model.set_time_steps_yearly,
            domain=pe.Reals,
            doc="total cost of energy system"
        )
        # NPV
        Variable.add_variable(
            model,
            name="NPV",
            index_sets = model.set_time_steps_yearly,
            domain=pe.Reals,
            doc="NPV of energy system"
        )

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <EnergySystem> """
        # get model
        model = cls.get_pyomo_model()

        # carbon emissions
        Constraint.add_constraint(
            model,
            name="constraint_carbon_emissions_total",
            index_sets = model.set_time_steps_yearly,
            rule=constraint_carbon_emissions_total_rule,
            doc="total carbon emissions of energy system"
        )
        # carbon emissions
        Constraint.add_constraint(
            model,
            name="constraint_carbon_emissions_cumulative",
            index_sets = model.set_time_steps_yearly,
            rule=constraint_carbon_emissions_cumulative_rule,
            doc="cumulative carbon emissions of energy system over time"
        )
        # cost of carbon emissions
        Constraint.add_constraint(
            model,
            name="constraint_carbon_cost_total",
            index_sets = model.set_time_steps_yearly,
            rule=constraint_carbon_cost_total_rule,
            doc="total carbon cost of energy system"
        )
        # carbon emissions
        Constraint.add_constraint(
            model,
            name="constraint_carbon_emissions_limit",
            index_sets = model.set_time_steps_yearly,
            rule=constraint_carbon_emissions_limit_rule,
            doc="limit of total carbon emissions of energy system"
        )
        # carbon emissions
        Constraint.add_constraint(
            model,
            name="constraint_carbon_emissions_budget",
            index_sets = model.set_time_steps_yearly,
            rule=constraint_carbon_emissions_budget_rule,
            doc="Budget of total carbon emissions of energy system"
        )
        # costs
        Constraint.add_constraint(
            model,
            name="constraint_cost_total",
            index_sets = model.set_time_steps_yearly,
            rule=constraint_cost_total_rule,
            doc="total cost of energy system"
        )
        # NPV
        Constraint.add_constraint(
            model,
            name="constraint_NPV",
            index_sets = model.set_time_steps_yearly,
            rule=constraint_NPV_rule,
            doc="NPV of energy system"
        )

    @classmethod
    def construct_objective(cls):
        """ constructs the pe.Objective of the class <EnergySystem> """
        logging.info("Construct pe.Objective")
        # get model
        model = cls.get_pyomo_model()

        # get selected objective rule
        if cls.get_analysis()["objective"] == "total_cost":
            objective_rule = objective_total_cost_rule
        elif cls.get_analysis()["objective"] == "total_carbon_emissions":
            objective_rule = objective_total_carbon_emissions_rule
        elif cls.get_analysis()["objective"] == "risk":
            logging.info("Objective of minimizing risk not yet implemented")
            objective_rule = objective_risk_rule
        else:
            raise KeyError(f"Objective type {cls.get_analysis()['objective']} not known")

        # get selected objective sense
        if cls.get_analysis()["sense"] == "minimize":
            objective_sense = pe.minimize
        elif cls.get_analysis()["sense"] == "maximize":
            objective_sense = pe.maximize
        else:
            raise KeyError(f"Objective sense {cls.get_analysis()['sense']} not known")

        # construct objective
        model.objective = pe.Objective(
            rule=objective_rule,
            sense=objective_sense
        )


def constraint_carbon_emissions_total_rule(model, year):
    """ add up all carbon emissions from technologies and carriers """
    return (
            model.carbon_emissions_total[year] ==
            # technologies
            model.carbon_emissions_technology_total[year]
            +
            # carriers
            model.carbon_emissions_carrier_total[year]
    )

def constraint_carbon_emissions_cumulative_rule(model, year):
    """ cumulative carbon emissions over time """
    # get parameter object
    params = Parameter.get_component_object()
    interval_between_years = EnergySystem.get_system()["intervalBetweenYears"]
    if year == model.set_time_steps_yearly.at(1):
        return (
                model.carbon_emissions_cumulative[year] ==
                model.carbon_emissions_total[year]
                + params.previous_carbon_emissions
        )
    else:
        return (
                model.carbon_emissions_cumulative[year] ==
                model.carbon_emissions_cumulative[year - 1]
                + model.carbon_emissions_total[year - 1] * (interval_between_years - 1)
                + model.carbon_emissions_total[year]
        )

def constraint_carbon_cost_total_rule(model, year):
    """ carbon cost associated with the carbon emissions of the system in each year """
    # get parameter object
    params = Parameter.get_component_object()
    return (
            model.cost_carbon_emissions_total[year] ==
            params.carbon_price[year] * model.carbon_emissions_total[year]
            # add overshoot price
            + model.carbon_emissions_overshoot[year] * params.carbon_price_overshoot
    )

def constraint_carbon_emissions_limit_rule(model, year):
    """ time dependent carbon emissions limit from technologies and carriers"""
    # get parameter object
    params = Parameter.get_component_object()
    if params.carbon_emissions_limit[year] != np.inf:
        return (
            params.carbon_emissions_limit[year] >= model.carbon_emissions_total[year]
        )
    else:
        return pe.Constraint.Skip

def constraint_carbon_emissions_budget_rule(model, year):
    """ carbon emissions budget of entire time horizon from technologies and carriers.
    The prediction extends until the end of the horizon, i.e.,
    last optimization time step plus the current carbon emissions until the end of the horizon """
    # get parameter object
    params = Parameter.get_component_object()
    interval_between_years = EnergySystem.get_system()["intervalBetweenYears"]
    if params.carbon_emissions_budget != np.inf: #TODO check for last year - without last term?
        return (
                params.carbon_emissions_budget + model.carbon_emissions_overshoot[year] >=
                model.carbon_emissions_cumulative[year] + model.carbon_emissions_total[year] * (interval_between_years - 1)
        )
    else:
        return pe.Constraint.Skip

def constraint_cost_total_rule(model, year):
    """ add up all costs from technologies and carriers"""
    return (
            model.cost_total[year] ==
            # capex
            model.capex_total[year] +
            # opex
            model.opex_total[year] +
            # carrier costs
            model.cost_carrier_total[year] +
            # carbon costs
            model.cost_carbon_emissions_total[year]
    )

def constraint_NPV_rule(model, year):
    """ discounts the annual capital flows to calculate the NPV """
    system = EnergySystem.get_system()
    discount_rate = EnergySystem.get_analysis()["discount_rate"]
    if system["optimized_years"] > 1:
        interval_between_years = system["intervalBetweenYears"]
    else:
        interval_between_years = 1

    return (
            model.NPV[year] ==
            model.cost_total[year] *
            sum(
                # economic discount
                ((1 / (1 + discount_rate)) ** (interval_between_years * (year - model.set_time_steps_yearly.at(1))+_intermediate_time_step))
                for _intermediate_time_step in range(0,interval_between_years)
            )
    )

# objective rules
def objective_total_cost_rule(model):
    """objective function to minimize the total cost"""
    system = EnergySystem.get_system()
    return (
        sum(
            model.NPV[year] *
            # discounted utility function
            ((1 / (1 + system["social_discount_rate"])) ** (
                        system["intervalBetweenYears"] * (year - model.set_time_steps_yearly.at(1))))
            for year in model.set_time_steps_yearly)
    )

def objectiveNPVRule(model):
    """ objective function to minimize NPV """
    return (
        sum(
            model.NPV[year]
            for year in model.set_time_steps_yearly)
    )

def objective_total_carbon_emissions_rule(model):
    """objective function to minimize total emissions"""
    return (sum(model.carbon_emissions_total[year] for year in model.set_time_steps_yearly))

def objective_risk_rule(model):
    """objective function to minimize total risk"""
    # TODO implement objective functions for risk
    return pe.Constraint.Skip

