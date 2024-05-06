"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Scenario settings settings.
==========================================================================================================================================================================="""
scenarios = dict()
# increase the demand of heat
scenarios["1"] = {"set_carriers": {"price_import": {"default_op": [1, 1.3, 1.6],
                                                    "default_op_fmt": "price_import_{}",  # increase price by 0, 30, 60%
                                                    'exclude': ['heat']}},  # exclude price increase for heat
                  'natural_gas_boiler': {'opex_specific_fixed': {'default': ['attributes_cheap', 'attributes_expensive']}}
                  }
