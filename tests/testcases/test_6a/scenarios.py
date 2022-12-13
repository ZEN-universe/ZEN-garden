"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Scenario settings settings.
==========================================================================================================================================================================="""
scenarios = dict()
scenarios["1"] = {"EnergySystem": ["carbonEmissionsBudget"]}        # change energy system parameter
scenarios["2"] = {"EnergySystem": ["carbonEmissionsLimit"]}         # change energy system parameter, yearly variation
scenarios["3"] = {"heat": ["demandCarrier"]}                        # change carrier attribute, intra-yearly variation
scenarios["4"] = {"heat": ["demandCarrier"]}                        # change carrier attribute, yearly variation (v1)
scenarios["5"] = {"heat": ["demandCarrierYearlyVariation"]}         # change carrier attribute, yearly variation (v2)
scenarios["6"] = {"natural_gas_boiler": ["existingCapacity"]}        # change technology attribute, add existing capacity
