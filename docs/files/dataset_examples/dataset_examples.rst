.. _dataset_examples:
################
Dataset Examples
################
ZEN-garden provides a number of small datasets to demonstrate the functionalities of ZEN-garden and to understand the data structure. The datasets are stored in the `datasets` directory of the ZEN-garden repository.
If you forked the ZEN-garden repository, you can find the datasets in the `documentation/dataset_examples` directory.
If you installed ZEN-garden using pip, you can download and execute the datasets with the `--example` flag (see :ref:`Run example`).

The following datasets are available:

1. `1_base_case`
2. `2_multi_year_optimization`
3. `3_reduced_import_availability`
4. `4_PWA_nonlinear_capex`
5. `5_multiple_time_steps_per_year`
6. `6_reduced_import_availability_yearly`
7. `7_time_series_aggregation`
8. `8_full_year`
9. `9_myopic_foresight`
10. `10_brown_field`
11. `11_multi_scenario`
12. `12_multiple_in_output_carriers_conversion_tech`
13. `13_yearly_interpolation`
14. `14_retrofitting_and_fuel_substitution`
15. `15_unit_consistency_expected_error`

1_base_case
-------------
Single year optimization with a simple energy system. Two nodes, one time step, two conversion, one storage, one transport technology.

2_multi_year_optimization
---------------------------
Same as `1_base_case`, but with multiple years.

3_reduced_import_availability
-------------------------------
Same as `2_multi_year_optimization`, but with variable heat demand in CH and DE, and reduced import availability in DE.

4_PWA_nonlinear_capex
------------------------
Same as `3_reduced_import_availability`, but with piecewise approximation of the capital cost function for natural gas boiler.

5_multiple_time_steps_per_year
--------------------------------
Same as `4_PWA_nonlinear_capex`, but with multiple time steps (96) per year.

6_reduced_import_availability_yearly
--------------------------------------
Same as `5_multiple_time_steps_per_year`, but with reduced import availability in CH for entire year. Also, introduction of heat pumps.

7_time_series_aggregation
---------------------------
Same as `6_reduced_import_availability_yearly`, but aggregation of the 96 time steps to 5 time steps.

8_yearly_variation
---------------------
Same as `7_time_series_aggregation`, but with yearly variation of the natural gas import price and electricity demand.

9_myopic_foresight
---------------------
Same as `8_yearly_variation`, but with myopic foresight.

10_brown_field
----------------
Same as `9_myopic_foresight`, but with brown field capacity expansion (existing capacities).

11_multi_scenario
-------------------
Same as `10_brown_field`, but with multiple scenarios. The different scenarios are different carbon prices.

12_multiple_in_output_carriers_conversion_tech
--------------------------------------------
Same as `11_multi_scenario`, but introduction of combined heat and power (CHP) technology (multiple output carriers) and of biogas as an additional input carrier for the CHP technology.

13_yearly_interpolation
-----------------------------
Same as `12_multiple_in_output_carriers_conversion_tech`, but with interpolation of missing values turned off.

14_retrofitting_and_fuel_substitution
-----------------------------
Same as `13_yearly_interpolation`, but with the introduction of a retrofitting technology (carbon capture plant).

15_unit_consistency_expected_error
------------------------------------
Same as `14_retrofitting_and_fuel_substitution`, but with an error in the unit consistency (expected failure) to show how ZEN-garden handles such errors.