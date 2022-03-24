"""=====================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      March-2022
Authors:      Alissa Ganter (aganter@ethz.ch)
              Johannes Burger (jburger@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
====================================================================================================================="""
from model import default_config

## Analysis - Default dictionary
analysis = default_config.analysis
## System - Default dictionary
system = default_config.system
## Solver - Default dictionary
solver = default_config.solver

## Analysis - settings update compared to default values
analysis["dataset"] = "NUTS0_KVAH_NL"
# analysis["modelFormulation"] = "KVAH_NL"  # Unnecessary ?!
analysis["objective"] = "TotalCost"  # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
# definition of the approximation
analysis["variablesNonlinearModel"] = {"builtCapacity": []}
analysis["nonlinearTechnologyApproximation"] = {"Capex": [], "ConverEfficiency": []}

## System - settings update compared to default values
system["setCarriers"] = ["carbon_liquid", "carbon_gaseous"]  # , "hydrogen", "biomass", "natural_gas"
system["setStorageTechnologies"] = []
system["setTransportTechnologies"] = ["carbon_truck"]  # hydrogen_truck_liquid ##, ,
system["setConversionTechnologies"] = ["carbon_liquefaction", "carbon_vaporizer"]
system["setTimeSteps"] = list(range(0, 1))
# if True, each element has its own time index; if False, use single time grid approach
system["multiGridTimeIndex"] = False
# default number of operational time steps, only used in single-grid time series aggregation
# TODO number of time steps per period = 1
system["numberTimeStepsDefault"] = 16
# system["setNodes"]
system["setNodes"] = ["CH", "IT", "NO"]

## Solver - settings update compared to default values
solver["gap"] = 0.01
solver["model"] = "MILP"
solver["verbosity"] = True
solver["performanceCheck"]["printDeltaIteration"] = 50
