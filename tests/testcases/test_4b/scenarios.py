"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Scenario settings settings.
==========================================================================================================================================================================="""
scenarios = dict()
# increase the demand of heat
scenarios["1"] = {"set_carriers": {"price_import": {"default_op": 1.2,
                                                    'exclude': ['heat']},
                                   'carbon_intensity_carrier_import': {'default_op': 2},
                                   'carbon_intensity_carrier_export': {'default_op': 2}},
                  'heat': {'carbon_intensity_carrier_import': {'default': 'attributes_1'},
                           'carbon_intensity_carrier_export': {'default': "attributes_1"}},  # overwrites the set carriers value
                  'set_technologies': {'capex_specific': {'default_op': 1.1}},  # increased capex by 10%
                  }
