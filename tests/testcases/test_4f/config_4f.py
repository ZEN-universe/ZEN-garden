"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from zen_garden.model import Config
import importlib.util
import os

# create a config
config = Config()
## Analysis - Default dictionary
analysis = config.analysis
## Solver - Default dictionary
solver = config.solver
## Scenarios - Default scenario dictionary
scenarios = config.scenarios

## Analysis - settings update compared to default values
analysis                                            = config.analysis         # get default settings from default config
analysis["dataset"]                                 = os.path.dirname(__file__)
analysis["objective"]                               = "TotalCost"                     # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
analysis["useExistingCapacities"]                   = True                           # use greenfield or brownfield approach
## Solver - settings update compared to default values
#
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

## System - load system configurations
from system_4f import system
config.system.update(system)
system  = config.system
## overwrite default system and scenario dictionaries
if system["conductScenarioAnalysis"]:
    assert os.path.exists(os.path.join(analysis["dataset"], "scenarios.py")), f"scenarios.py is missing from selected dataset '{analysis['dataset']}"
    from scenarios import scenarios
    config.scenarios.update(scenarios)
