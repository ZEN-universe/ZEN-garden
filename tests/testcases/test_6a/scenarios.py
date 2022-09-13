"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Scenario settings settings.
==========================================================================================================================================================================="""
from model.default_config   import scenarios
import numpy                as np

# scenario analysis
scenarios["2"]            = {"EnergySystem": ["carbonEmissionsLimit"]}
scenarios["3"]            = {"natural_gas": ["demandCarrier"]}
