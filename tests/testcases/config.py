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
analysis["objective"] = "total_cost"
# use greenfield or brownfield approach
analysis["use_capacities_existing"] = True

## Solver - settings update compared to default values
solver["name"] = "glpk" # free solver
# solver["solver_options"]["Method"] = 2
# solver["solver_options"]["NodeMethod"] = 2
solver["solver_options"]["BarHomogeneous"] = 1
# solver["solver_options"]["Presolve"] = -1
solver["solver_options"]["Threads"] = 46
# solver["solver_options"]["CrossoverBasis"] = 0
# solver["solver_options"]["Crossover"] = 0
solver["solver_options"]["ScaleFlag"] = 2
solver["analyze_numerics"] = True
solver["immutable_unit"] = ["hour","km"]

