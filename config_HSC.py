"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""
from model import default_config

## Analysis - Defaul dictionary
analysis = default_config.analysis
## System - Defaul dictionary
system = default_config.system
## Solver - Defaul dictionary
solver = default_config.solver   

## Analysis - settings update compared to default values
analysis["dataset"]                                 = "NUTS0_HSC"
analysis["modelFormulation"]                        = "HSC"
analysis["objective"]                               = "TotalCost" # choose from "TotalCost", "TotalCarbonEmissions", "Risk"
# definition of the approximation
analysis["variablesNonlinearModel"]                 = {"builtCapacity": []}
analysis["nonlinearTechnologyApproximation"]        = {"Capex": [], "ConverEfficiency":[]}

## System - settings update compared to default values
system["setCarriers"]                               = ["electricity", "hydrogen", "natural_gas", "biomass"] #
system["setStorageTechnologies"]                    = []
system["setTransportTechnologies"]                  = ["hydrogen_pipeline"]# , "hydrogen_truck_gas", "hydrogen_train","hydrogen_ship",
system["setConversionTechnologies"]                 = ["electrolysis", "SMR", "biomass_gasifier", "biomass_SMR"] #
system["setTimeSteps"]                              = list(range(0,15))
system["multiGridTimeIndex"]                        = False # if True, each element has its own time index; if False, use single time grid approach
system["numberTimeStepsDefault"]                    = 16 # default number of operational time steps, only used in single-grid time series aggregation # TODO number of time steps per period = 1
system["setNodes"]                                  = ["IT","DE","AT"]#, "IT","FR"]
#system["setNodes"]                                  = ["AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
                                                       #"FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "ME", "MK",
                                                       #"MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK", "UK"]

## Solver - settings update compared to default values
solver["gap"]                                       = 0.01
solver["model"]                                     = "MILP"
solver["verbosity"]                                 = True
solver["performanceCheck"]["printDeltaIteration"]   = 50