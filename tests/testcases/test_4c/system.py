"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""

## System - Default dictionary
system = dict()

## System - settings update compared to default values
system['setConversionTechnologies']     = ["natural_gas_boiler"]
system['setStorageTechnologies']        = ["natural_gas_storage"]
system['setTransportTechnologies']      = ["natural_gas_pipeline"]

system['setNodes']                      = ["DE","CH"]
system["socialDiscountRate"]            = 0     # similar to discount factor, but for discounted utility model
system["knowledgeSpilloverRate"]        = 0.025
# time steps
system["referenceYear"]                 = 2022
system["unaggregatedTimeStepsPerYear"]  = 1
system["aggregatedTimeStepsPerYear"]    = 1
system["conductTimeSeriesAggregation"]  = False

system["optimizedYears"]                = 3
system["intervalBetweenYears"]          = 1
system["useRollingHorizon"]             = False
system["yearsInRollingHorizon"]         = 1

