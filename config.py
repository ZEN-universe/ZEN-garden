"""
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
"""
from zen_garden.model import Config
import importlib.util
import os

config = Config()
## Analysis - Default dictionary
analysis = config.analysis
## Solver - Default dictionary
solver = config.solver
## Scenarios - Default scenario dictionary
scenarios = config.scenarios

## Analysis - settings update compared to default values
analysis["dataset"] = os.path.join(os.path.dirname(__file__), "data/county_CA_0507_288_3") # path to dataset

analysis["objective"] =  "total_cost" #"total_carbon_emissions" total_cost
# use greenfield or brownfield approach depending on dataset name
analysis['use_capacities_existing'] = True # False
analysis["folder_output"] = "./outputs/"

## Timeseries Aggregation
analysis["time_series_aggregation"]["noSegments"] = 156 #288 # 12 days times 24 hours
analysis["time_series_aggregation"]["segmentation"] = True
analysis["time_series_aggregation"]["clusterMethod"] = "k_means"
analysis["time_series_aggregation"]["rescaleClusterPeriods"] = True
# analysis["time_series_aggregation"]["representationMethod"] = "durationRepresentation"

## Solver - settings update compared to default values
solver["name"] = "gurobi" # either free solver 'glpk' or 'gurobi'
solver["solver_options"]["Method"] = -1
#solver["solver_options"]["NumericFocus"] = 1
# solver["solver_options"]["NodeMethod"]   = 2
solver["solver_options"]["BarHomogeneous"]   = 1
# solver["solver_options"]["Presolve"]     = -1
solver["solver_options"]["Threads"]      = 6
# solver["solver_options"]["CrossoverBasis"]   = 0
# solver["solver_options"]["Crossover"]    = 0
solver["solver_options"]["DualReductions"] = 0
solver["solver_options"]["ScaleFlag"]    = 2
solver["analyze_numerics"]               = True
solver["immutable_unit"]                 = ["hour","km"]
solver["use_symbolic_labels"] = False
solver['symbolic_solver_labels'] = True
solver["add_duals"] = True
solver["check_unit_consistency"] = True
solver["recommend_base_units"] = True