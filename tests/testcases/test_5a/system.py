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
system['set_conversion_technologies']     = ["natural_gas_boiler"] #, "natural_gas_expander"
system['set_conditioning_technologies']   = ["natural_gas_compressor"]
system['set_storage_technologies']        = ["natural_gas_storage"]
system['set_transport_technologies']      = ["natural_gas_pipeline"]

system['set_nodes']                      = ["DE","CH"]
system["social_discount_rate"]            = 0     # similar to discount factor, but for discounted utility model
system["knowledgeSpilloverRate"]        = 0.025
# time steps
system["referenceYear"]                 = 2022
system["unaggregated_time_steps_per_year"]  = 1
system["aggregatedTimeStepsPerYear"]    = 1
system["conduct_time_series_aggregation"]  = False

system["optimized_years"]                = 1
system["intervalBetweenYears"]          = 1
system["useRollingHorizon"]             = False
system["yearsInRollingHorizon"]         = 1

