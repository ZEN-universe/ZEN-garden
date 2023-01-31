"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        March-2022
Authors:        Alissa Ganter (aganter@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints of the conditioning technologies.
                The class takes the abstract optimization model as an input, and adds parameters, variables and
                constraints of the conversion technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import pandas as pd
from ..energy_system import EnergySystem
from .conversion_technology import ConversionTechnology


class ConditioningTechnology(ConversionTechnology):
    # set label
    label = "set_conditioning_technologies"

    def __init__(self, tech, optimization_setup):
        """init conditioning technology object
        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of """

        logging.info(f'Initialize conditioning technology {tech}')
        super().__init__(tech, optimization_setup)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        self.add_conditioning_carriers()

    def add_conditioning_carriers(self):
        """add conditioning carriers to system"""
        subset = "set_conditioning_carriers"
        analysis = self.optimization_setup.analysis
        system = self.optimization_setup.system
        # add set_conditioning_carriers to analysis and indexing_sets
        if subset not in analysis["subsets"]["set_carriers"]:
            analysis["subsets"]["set_carriers"].append(subset)
        # add set_conditioning_carriers to system
        if subset not in system.keys():
            system[subset] = []
        if self.output_carrier[0] not in system[subset]:
            system[subset] += self.output_carrier

    def get_conver_efficiency(self):
        """retrieves and stores conver_efficiency for <ConditioningTechnology>.
        Create dictionary with input parameters with the same format as pwa_conver_efficiency"""
        set_time_steps_yearly = self.energy_system.set_time_steps_yearly
        specific_heat = self.data_input.extract_attribute("specific_heat")["value"]
        specific_heat_ratio = self.data_input.extract_attribute("specific_heat_ratio")["value"]
        pressure_in = self.data_input.extract_attribute("pressure_in")["value"]
        pressure_out = self.data_input.extract_attribute("pressure_out")["value"]
        temperature_in = self.data_input.extract_attribute("temperature_in")["value"]
        isentropic_efficiency = self.data_input.extract_attribute("isentropic_efficiency")["value"]

        # calculate energy consumption
        _pressure_ratio = pressure_out / pressure_in
        _exponent = (specific_heat_ratio - 1) / specific_heat_ratio
        if self.data_input.exists_attribute("lower_heating_value", column=None):
            _lower_heating_value = self.data_input.extract_attribute("lower_heating_value")["value"]
            specific_heat = specific_heat / _lower_heating_value
        _energy_consumption = specific_heat * temperature_in / isentropic_efficiency * (_pressure_ratio ** _exponent - 1)

        # check input and output carriers
        _input_carriers = self.input_carrier.copy()
        if self.reference_carrier[0] in _input_carriers:
            _input_carriers.remove(self.reference_carrier[0])
        assert len(_input_carriers) == 1, f"{self.name} can only have 1 input carrier besides the reference carrier."
        assert len(self.output_carrier) == 1, f"{self.name} can only have 1 output carrier."
        # create dictionary
        self.conver_efficiency_is_pwa = False
        self.conver_efficiency_linear = dict()
        self.conver_efficiency_linear[self.output_carrier[0]] = \
        self.data_input.create_default_output(index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly, manual_default_value=1)[
            0]  # TODO losses are not yet accounted for
        self.conver_efficiency_linear[_input_carriers[0]] = \
        self.data_input.create_default_output(index_sets=["set_nodes", "set_time_steps_yearly"], time_steps=set_time_steps_yearly, manual_default_value=_energy_consumption)[0]
        # dict to dataframe
        self.conver_efficiency_linear = pd.DataFrame.from_dict(self.conver_efficiency_linear)
        self.conver_efficiency_linear.columns.name = "carrier"
        self.conver_efficiency_linear = self.conver_efficiency_linear.stack()
        _conver_efficiency_levels = [self.conver_efficiency_linear.index.names[-1]] + self.conver_efficiency_linear.index.names[:-1]
        self.conver_efficiency_linear = self.conver_efficiency_linear.reorder_levels(_conver_efficiency_levels)

    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <ConditioningTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        # get parent carriers
        _output_carriers = optimization_setup.get_attribute_of_all_elements(cls, "output_carrier")
        _reference_carriers = optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier")
        _parent_carriers = list()
        _child_carriers = dict()
        for tech, carrier_ref in _reference_carriers.items():
            if carrier_ref[0] not in _parent_carriers:
                _parent_carriers += carrier_ref
                _child_carriers[carrier_ref[0]] = list()
            if _output_carriers[tech] not in _child_carriers[carrier_ref[0]]:
                _child_carriers[carrier_ref[0]] += _output_carriers[tech]
                _conditioning_carriers = list()
        _conditioning_carriers = _parent_carriers + [carrier[0] for carrier in _child_carriers.values()]

        # update indexing sets
        optimization_setup.energy_system.indexing_sets.append("set_conditioning_carriers")
        optimization_setup.energy_system.indexing_sets.append("set_conditioning_carrier_parents")

        # set of conditioning carriers
        model.set_conditioning_carriers = pe.Set(initialize=_conditioning_carriers, doc="set of conditioning carriers")
        # set of parent carriers
        model.set_conditioning_carrier_parents = pe.Set(initialize=_parent_carriers, doc="set of parent carriers of conditioning")
        # set that maps parent and child carriers
        model.set_conditioning_carrier_children = pe.Set(model.set_conditioning_carrier_parents, initialize=_child_carriers,
            doc="set of child carriers associated with parent carrier used in conditioning")
