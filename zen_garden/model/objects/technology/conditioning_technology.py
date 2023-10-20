"""
:Title:          ZEN-GARDEN
:Created:        March-2022
:Authors:        Alissa Ganter (aganter@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables and constraints of the conditioning technologies.
The class takes the abstract optimization model as an input, and adds parameters, variables and
constraints of the conversion technologies.
"""
import logging

import pandas as pd

from .conversion_technology import ConversionTechnology


class ConditioningTechnology(ConversionTechnology):
    # set label
    label = "set_conditioning_technologies"

    def __init__(self, tech, optimization_setup):
        """init conditioning technology object

        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of """
        super().__init__(tech, optimization_setup)

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

    def get_conversion_factor(self):
        """retrieves and stores conversion_factor for <ConditioningTechnology>.
        Create dictionary with input parameters with the same format as pwa_conversion_factor"""
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
        self.conversion_factor_is_pwa = False
        conversion_factor = dict()
        conversion_factor[self.output_carrier[0]] = \
        self.data_input.create_default_output(index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", manual_default_value=1)[
            0]  # TODO losses are not yet accounted for
        conversion_factor[_input_carriers[0]] = \
        self.data_input.create_default_output(index_sets=["set_nodes", "set_time_steps"], time_steps="set_base_time_steps_yearly", manual_default_value=_energy_consumption)[0]
        # dict to dataframe
        conversion_factor = pd.DataFrame.from_dict(conversion_factor)
        conversion_factor.columns.name = "carrier"
        conversion_factor = conversion_factor.stack()
        conversion_factor_levels = [conversion_factor.index.names[-1]] + conversion_factor.index.names[:-1]
        self.raw_time_series["conversion_factor"] = conversion_factor.reorder_levels(conversion_factor_levels)

    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <ConditioningTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
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
        _conditioning_carriers = _parent_carriers + [carrier for carriers in _child_carriers.values() for carrier in carriers]

        # update indexing sets
        optimization_setup.energy_system.indexing_sets.append("set_conditioning_carriers")
        optimization_setup.energy_system.indexing_sets.append("set_conditioning_carrier_parents")

        # set of conditioning carriers
        optimization_setup.sets.add_set(name="set_conditioning_carriers", data=_conditioning_carriers, doc="set of conditioning carriers")
        # set of parent carriers
        optimization_setup.sets.add_set(name="set_conditioning_carrier_parents", data=_parent_carriers, doc="set of parent carriers of conditioning")
        # set that maps parent and child carriers
        optimization_setup.sets.add_set(name="set_conditioning_carrier_children", data=_child_carriers, doc="set of child carriers associated with parent carrier used in conditioning",
                                        index_set="set_conditioning_carrier_parents")
