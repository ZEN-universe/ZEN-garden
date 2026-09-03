# Creating datasets from the examples

The datasets in this directory demonstrate ZEN-garden features with small,
incremental energy-system models. Start with
[`1_base_case`](./1_base_case/) when creating a dataset and copy only the
features needed from the later examples.

Every dataset contains:

- `system.yaml`, which selects the nodes and technologies and configures the
  optimization;
- `energy_system`, which defines nodes, edges, units, and system-wide input
  data;
- `set_carriers`, which defines the energy carriers and their input data; and
- `set_technologies`, which defines conversion, storage, transport, and, where
  applicable, retrofit technologies.

Values shared by all indices are stored in `attributes.yaml`. Values that vary
by node, edge, time step, or year are stored in parameter-specific CSV files.
The following examples show the currently supported dataset sequence.

## Dataset sequence

1. [`1_base_case`](./1_base_case/): basic energy-system structure
2. [`2_multi_year_optimization`](./2_multi_year_optimization/): multiple
   optimization years
3. [`3_reduced_import_availability`](./3_reduced_import_availability/):
   node-specific import limits and transport investment
4. [`4_multiple_time_steps_per_year`](./4_multiple_time_steps_per_year/):
   multiple operational time steps
5. [`5_reduced_import_availability_yearly`](./5_reduced_import_availability_yearly/):
   yearly import limits and storage investment
6. [`6_time_series_aggregation`](./6_time_series_aggregation/): representative
   time steps
7. [`7_yearly_variation`](./7_yearly_variation/): yearly scaling of time series
8. [`8_myopic_foresight`](./8_myopic_foresight/): rolling-horizon optimization
9. [`9_brown_field`](./9_brown_field/): existing technology capacities
10. [`10_multi_scenario`](./10_multi_scenario/): scenario analysis
11. [`11_multiple_in_output_carriers_conversion`](./11_multiple_in_output_carriers_conversion/):
    conversion technologies with multiple input and output carriers
12. [`12_yearly_interpolation`](./12_yearly_interpolation/): interpolation of
    missing yearly values
13. [`13_retrofitting_and_fuel_substitution`](./13_retrofitting_and_fuel_substitution/):
    retrofit technologies and fuel substitution
14. [`14_unit_consistency_expected_error`](./14_unit_consistency_expected_error/):
    intentional unit errors for testing input validation

## 1_base_case

This example supplies heat and electricity in Switzerland (`CH`) and Germany
(`DE`). Natural gas boilers and photovoltaics supply the demands. A natural gas
storage and pipeline are available but are not needed because natural gas can
be imported without limits at both nodes.

Use this example as the template for a new dataset. Adjust the selected nodes
and technologies in `system.yaml`, then add or remove the corresponding carrier
and technology directories.

## 2_multi_year_optimization

This example extends the base case to multiple years. `optimized_years` defines
the number of years and `interval_between_years` defines the spacing between
them. The heat demand varies by year through the heat carrier's `demand.csv`.

## 3_reduced_import_availability

The file `availability_import.csv` limits natural gas imports in Germany. The
model therefore invests in a pipeline from Switzerland. The example also
enables `double_capex_transport`, which separates distance-dependent transport
costs from capacity-dependent costs.

## 4_multiple_time_steps_per_year

This example introduces operation within a year. Both
`aggregated_time_steps_per_year` and `unaggregated_time_steps_per_year` are set
to 96, and the heat and electricity demands are provided at hourly resolution.

## 5_reduced_import_availability_yearly

The file `availability_import_yearly.csv` reduces Switzerland's annual natural
gas import availability over time. Unlike `availability_import.csv`, it limits
the total import in a year. Storage and pipeline investments can compensate for
the annual restrictions.

## 6_time_series_aggregation

This example enables `conduct_time_series_aggregation` and aggregates 96 input
time steps to 10 representative time steps. Its yearly import-availability CSV
also demonstrates the supported layout with years in columns and nodes in rows.

## 7_yearly_variation

Yearly-variation files scale a time series by one factor per year. This example
varies the natural gas import price and electricity demand. It supplies 8760
unaggregated hourly time steps and uses time-series aggregation to optimize 10
representative steps.

## 8_myopic_foresight

This example enables `use_rolling_horizon`. The
`years_in_rolling_horizon` setting limits how many future years are visible in
each optimization step; it is set to one here.

## 9_brown_field

The brown-field example adds existing photovoltaic capacities with
`existing_capacities.csv`. Existing assets can have been built before the
optimization horizon or can represent committed future construction.

## 10_multi_scenario

This example enables `conduct_scenario_analysis` and defines its scenarios in
`scenarios.yaml`. The scenarios select alternative input files, including
different carbon prices and carrier attributes, without duplicating the full
dataset.

## 11_multiple_in_output_carriers_conversion

The combined heat and power plant consumes natural gas and biogas and produces
heat and electricity. Its `conversion_factor` values relate every input and
output carrier to the reference carrier.

## 12_yearly_interpolation

ZEN-garden linearly interpolates missing yearly parameter values by default.
This example demonstrates that behavior for annual carbon-emission limits and
carbon prices. Parameters listed in
`energy_system/parameters_interpolation_off.yaml` use their default value
instead of interpolation.

## 13_retrofitting_and_fuel_substitution

This example adds retrofit technologies to a combined heat and power plant.
The retrofit options substitute e-fuel for natural gas or capture carbon for
permanent storage. `retrofit_flow_coupling_factor` connects each retrofit
technology to its base conversion technology.

## 14_unit_consistency_expected_error

This is intentionally an invalid dataset. It mixes energy- and mass-based units
for natural gas and the natural gas pipeline so that ZEN-garden's unit checks
raise an error. Use it to understand validation messages, not as a template for
a valid dataset.
