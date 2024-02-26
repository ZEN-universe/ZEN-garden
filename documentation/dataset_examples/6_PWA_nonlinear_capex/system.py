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
system['set_conversion_technologies']     = ["natural_gas_boiler", "photovoltaics", "heat_pump"]
system['set_storage_technologies']        = ["natural_gas_storage"]
system['set_transport_technologies']      = ["natural_gas_pipeline"]

system['set_nodes']                      = ["DE", "CH"]

# time steps
system["reference_year"]                 = 2023
system["unaggregated_time_steps_per_year"]  = 1
system["aggregated_time_steps_per_year"]    = 1
system["conduct_time_series_aggregation"]  = False

system["optimized_years"]                = 3
system["interval_between_years"]          = 1
system["use_rolling_horizon"]             = False
system["years_in_rolling_horizon"]         = 1
