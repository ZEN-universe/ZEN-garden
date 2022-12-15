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
import pyomo.environ                                as pe
import pandas                                       as pd
import numpy                                        as np
from ..energy_system                    import EnergySystem
from .conversion_technology import ConversionTechnology

class ConditioningTechnology(ConversionTechnology):
    # set label
    label = "setConditioningTechnologies"
    # empty list of elements
    list_of_elements = []

    def __init__(self, tech):
        """init conditioning technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize conditioning technology {tech}')
        super().__init__(tech)
        # store input data
        self.store_input_data()
        # add ConversionTechnology to list
        ConditioningTechnology.add_element(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        self.setConditioningCarriers()

    def setConditioningCarriers(self):
        """add conditioning carriers to system"""
        subset   = "setConditioningCarriers"
        analysis = EnergySystem.get_analysis()
        system   = EnergySystem.get_system()
        # add setConditioningCarriers to analysis and indexing_sets
        if subset not in analysis["subsets"]["setCarriers"]:
            analysis["subsets"]["setCarriers"].append(subset)
        # add setConditioningCarriers to system
        if subset not in system.keys():
            system[subset] = []
        if self.outputCarrier[0] not in system[subset]:
            system[subset] += self.outputCarrier

    def getConverEfficiency(self):
        """retrieves and stores converEfficiency for <ConditioningTechnology>.
        Create dictionary with input parameters with the same format as PWAConverEfficiency"""
        set_time_steps_yearly          = EnergySystem.get_energy_system().set_time_steps_yearly
        self.specificHeat         = self.datainput.extract_attribute("specificHeat")["value"]
        self.specificHeatRatio    = self.datainput.extract_attribute("specificHeatRatio")["value"]
        self.pressureIn           = self.datainput.extract_attribute("pressureIn")["value"]
        self.pressureOut          = self.datainput.extract_attribute("pressureOut")["value"]
        self.temperatureIn        = self.datainput.extract_attribute("temperatureIn")["value"]
        self.isentropicEfficiency = self.datainput.extract_attribute("isentropicEfficiency")["value"]

        # calculate energy consumption
        _pressureRatio     = self.pressureOut / self.pressureIn
        _exponent          = (self.specificHeatRatio - 1) / self.specificHeatRatio
        if self.datainput.exists_attribute("lowerHeatingValue", column=None):
            _lowerHeatingValue = self.datainput.extract_attribute("lowerHeatingValue")["value"]
            self.specificHeat  = self.specificHeat / _lowerHeatingValue
        _energyConsumption = self.specificHeat * self.temperatureIn / self.isentropicEfficiency \
                            * (_pressureRatio ** _exponent - 1)

        # check input and output carriers
        _inputCarriers = self.inputCarrier.copy()
        if self.referenceCarrier[0] in _inputCarriers:
            _inputCarriers.remove(self.referenceCarrier[0])
        assert len(_inputCarriers) == 1, f"{self.name} can only have 1 input carrier besides the reference carrier."
        assert len(self.outputCarrier) == 1, f"{self.name} can only have 1 output carrier."
        # create dictionary
        self.converEfficiencyIsPWA                            = False
        self.conver_efficiency_linear                           = dict()
        self.conver_efficiency_linear[self.outputCarrier[0]]    = self.datainput.create_default_output(index_sets=["setNodes","set_time_steps"],
                                                                                                   column=None,
                                                                                                   time_steps=set_time_steps_yearly,
                                                                                                   manual_default_value = 1)[0] # TODO losses are not yet accounted for
        self.conver_efficiency_linear[_inputCarriers[0]]        = self.datainput.create_default_output(index_sets=["setNodes", "set_time_steps"],
                                                                                                   column=None,
                                                                                                   time_steps=set_time_steps_yearly,
                                                                                                   manual_default_value=_energyConsumption)[0]
        # dict to dataframe
        self.conver_efficiency_linear              = pd.DataFrame.from_dict(self.conver_efficiency_linear)
        self.conver_efficiency_linear.columns.name = "carrier"
        self.conver_efficiency_linear              = self.conver_efficiency_linear.stack()
        _conver_efficiency_levels                  = [self.conver_efficiency_linear.index.names[-1]] + self.conver_efficiency_linear.index.names[:-1]
        self.conver_efficiency_linear              = self.conver_efficiency_linear.reorder_levels(_conver_efficiency_levels)

    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <ConditioningTechnology> """
        model = EnergySystem.get_pyomo_model()
        # get parent carriers
        _outputCarriers    = cls.get_attribute_of_all_elements("outputCarrier")
        _referenceCarriers = cls.get_attribute_of_all_elements("referenceCarrier")
        _parentCarriers    = list()
        _childCarriers     = dict()
        for tech, carrierRef in _referenceCarriers.items():
            if carrierRef[0] not in _parentCarriers:
                _parentCarriers += carrierRef
                _childCarriers[carrierRef[0]] = list()
            if _outputCarriers[tech] not in _childCarriers[carrierRef[0]]:
                _childCarriers[carrierRef[0]] +=_outputCarriers[tech]
                _conditioningCarriers = list()
        _conditioningCarriers = _parentCarriers+[carrier[0] for carrier in _childCarriers.values()]

        # update indexing sets
        EnergySystem.indexing_sets.append("setConditioningCarriers")
        EnergySystem.indexing_sets.append("setConditioningCarrierParents")

        # set of conditioning carriers
        model.setConditioningCarriers = pe.Set(
            initialize=_conditioningCarriers,
            doc="set of conditioning carriers. Dimensions: setConditioningCarriers"
        )
        # set of parent carriers
        model.setConditioningCarrierParents = pe.Set(
            initialize=_parentCarriers,
            doc="set of parent carriers of conditioning. Dimensions: setConditioningCarrierParents"
        )
        # set that maps parent and child carriers
        model.setConditioningCarrierChildren = pe.Set(
            model.setConditioningCarrierParents,
            initialize=_childCarriers,
            doc="set of child carriers associated with parent carrier used in conditioning. Dimensions: setConditioningCarrierChildren"
        )
