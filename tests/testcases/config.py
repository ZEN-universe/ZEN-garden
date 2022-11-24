"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""

from zen_garden.model import Config
import os

# create a config
config = Config()

## Analysis - Default dictionary
analysis = config.analysis
## Solver - Default dictionary
solver = config.solver

## Analysis - settings update compared to default values
analysis["objective"]                               = "TotalCost"                     # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
analysis["useExistingCapacities"]                   = True                           # use greenfield or brownfield approach

## Solver - settings update compared to default values
solver["name"] = "glpk" # free solver
# solver["solverOptions"]["Method"]       = 2
# solver["solverOptions"]["NodeMethod"]   = 2
solver["solverOptions"]["BarHomogeneous"]   = 1
# solver["solverOptions"]["Presolve"]     = -1
solver["solverOptions"]["Threads"]      = 46
# solver["solverOptions"]["CrossoverBasis"]   = 0
# solver["solverOptions"]["Crossover"]    = 0
solver["solverOptions"]["ScaleFlag"]    = 2
solver["analyzeNumerics"]               = True
solver["immutableUnit"]                 = ["hour","km"]

