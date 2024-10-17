################
Dataset Examples
################
ZEN-garden provides a number of small datasets to demonstrate the functionalities of ZEN-garden and to understand the data structure. The datasets are stored in the `datasets` directory of the ZEN-garden repository.
If you forked the ZEN-garden repository, you can find the datasets in the `documentation/dataset_examples` directory.
If you installed ZEN-garden using pip, you can download and execute the datasets with the `--example` flag (see :ref:`Run example`).

The following datasets are available:
1. `1_base_case`
2. `2_add_photovoltaics`
3. `3_multi_year_optimization`
4. `4_variable_demand`
5. `5_reduced_import_availability`
6. `6_PWA_nonlinear_capex`
7. `7_multiple_time_steps_per_year`
8. `8_reduced_import_availability_yearly`
9. `9_time_series_aggregation`
10. `10_full_year`
11. `11_yearly_variation`
12. `12_myopic_foresight`
13. `13_brown_field`
14. `14_multi_scenario`
15. `15_multiple_output_carriers_conversion_tech`
16. `16_multiple_input_carriers_conversion_tech`
17. `17_yearly_parameter_missing_values`
18. `18_yearly_interpolation_off`
19. `19_retrofitting_technologies`
20. `20_unit_consistency_expectedfailure`

1_base_case
-------------
Single year optimization with a simple energy system. Two nodes, one time step, one conversion, one storage, one transport technology.

2_add_photovoltaics
---------------------
Same as `1_base_case`, but with photovoltaics added.

3_multi_year_optimization
---------------------------
Same as `2_add_photovoltaics`, but with multiple years.

4_variable_demand
-------------------
Same as `3_multi_year_optimization`, but with larger heat demand in DE than in CH.

5_reduced_import_availability
-------------------------------
Same as `4_variable_demand`, but with reduced import availability in DE.

6_PWA_nonlinear_capex
------------------------
Same as `5_reduced_import_availability`, but with piecewise approximation of the capital cost function for natural gas boiler.

7_multiple_time_steps_per_year
--------------------------------
Same as `6_PWA_nonlinear_capex`, but with multiple time steps (96) per year.

8_reduced_import_availability_yearly
--------------------------------------
Same as `7_multiple_time_steps_per_year`, but with reduced import availability in CH for entire year. Also, introduction of heat pumps.

9_time_series_aggregation
---------------------------
Same as `8_reduced_import_availability_yearly`, but aggregation of the 96 time steps to 5 time steps.

10_full_year
--------------
Same as `9_time_series_aggregation`, but with a full year of data, aggregated to 10 representative time steps.

11_yearly_variation
---------------------
Same as `10_full_year`, but with yearly variation of the natural gas import price and electricity demand.

12_myopic_foresight
---------------------
Same as `11_yearly_variation`, but with myopic foresight.

13_brown_field
----------------
Same as `12_myopic_foresight`, but with brown field capacity expansion (existing capacities).

14_multi_scenario
-------------------
Same as `13_brown_field`, but with multiple scenarios. The different scenarios are different carbon prices.

15_multiple_output_carriers_conversion_tech
---------------------------------------------
Same as `14_multi_scenario`, but introduction of combined heat and power (CHP) technology (multiple output carriers).

16_multiple_input_carriers_conversion_tech
--------------------------------------------
Same as `15_multiple_output_carriers_conversion_tech`, but introduction of biogas as an additional input carrier for the CHP technology.

17_yearly_parameter_missing_values
------------------------------------
Same as `16_multiple_input_carriers_conversion_tech`, but with missing values in the carbon emission limit to show interpolation of missing values.

18_yearly_interpolation_off
-----------------------------
Same as `17_yearly_parameter_missing_values`, but with interpolation of missing values turned off.

19_retrofitting_technologies
-----------------------------
Same as `18_yearly_interpolation_off`, but with the introduction of a retrofitting technology (carbon capture plant).

20_unit_consistency_expectedfailure
------------------------------------
Same as `19_retrofitting_technologies`, but with an error in the unit consistency (expected failure) to show how ZEN-garden handles such errors.