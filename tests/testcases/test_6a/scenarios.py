"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Scenario settings settings.
==========================================================================================================================================================================="""
scenarios = dict()
scenarios["1"] = {"EnergySystem": ["carbon_emissions_budget"]}        # change energy system parameter
scenarios["2"] = {"EnergySystem": ["carbon_emissions_limit"]}         # change energy system parameter, yearly variation
scenarios["3"] = {"heat": ["demand"]}                        # change carrier attribute, intra-yearly variation
scenarios["4"] = {"heat": ["demand"]}                        # change carrier attribute, yearly variation (v1)
scenarios["5"] = {"heat": ["demand_yearly_variation"]}         # change carrier attribute, yearly variation (v2)
scenarios["6"] = {"natural_gas_boiler": ["capacity_existing"]}        # change technology attribute, add existing capacity
