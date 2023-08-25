"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Scenario settings settings.
==========================================================================================================================================================================="""
scenarios = dict()
scenarios["1"] = {"natural_gas": {"price_import": {"default": "attributes_1"}},  # new import price for natural gas
                  'natural_gas_boiler': {'capex_specific': {'default_op': 1.1}},  # increased capex by 10%
                  'heat': {'demand': {'file': 'demand_1',  # new demand file for heat
                                      'file_op': 2}},  # doubles the demand
                  "EnergySystem": {"price_carbon_emissions":{"default": "attributes_1"}} # increase the price for carbon emissions
                  }

