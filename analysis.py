"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config

## Analysis - Defaul dictionary
analysis = default_config.analysis
## Solver - Defaul dictionary
solver = default_config.solver

## Analysis - settings update compared to default values
analysis["dataset"]                                 = "NUTS0_HSC"
analysis["objective"]                               = "TotalCost"                     # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
analysis["subsets"]["setConversionTechnologies"]    = ["setConditioningTechnologies"] # ConditioningTechnologies are a special case of ConverstionTechnologies

## Solver - settings update compared to default values
solver["gap"]                                       = 0.01
solver["model"]                                     = "MILP"
solver["verbosity"]                                 = True
solver["performanceCheck"]["printDeltaIteration"]   = 50