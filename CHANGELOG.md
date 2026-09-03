# Changelog

This file gets automatically updated in ZEN-garden's continuous integration 
procedures. Do not edit the file manually.

## [v3.0.0] - 2026-09-03 

### New Features ✨
- Add legacy scenario readers for results versions v1, v2, and v3 to preserve backward compatibility with results produced by earlier ZEN-garden versions. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Add Leontief input-output tables for sectoral cost and emission allocation, exposed as `Results.get_sectoral_costs()` and `get_sectoral_emissions()`, with on-disk caching and optional spatial resolution. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Allow config, system, scenario, attribute, and unit input files to be specified as YAML in addition to JSON, improving readability and enabling comments. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Add a plugin event fired after model-schema creation so that plugins can register additional parameters, variables, expressions, and constraints before the model is built. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Set up `uv` to manage Python dependencies, making it easier to keep environments consistent across branches with differing dependency versions. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Speed up and reduce the memory footprint of results reading by loading index names without reading all data, caching xarray datasets, skipping re-validation of pydantic JSON files, and constructing `get_full_ts` dataframes more efficiently. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Add the missing input-data header for retrofitting technologies. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]

### Bug Fixes 🐛
- Adjust the technology diffusion-limit constraint to exclude the target technology from its own unbounded market-share allowance, deriving the allowance only from other technologies in the same class with the same reference carrier. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Preserve indexed attribute dimensions and resolve fallback units consistently, preventing single-directed transport data from being squeezed incorrectly during downstream processing. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Correct a working-directory-dependent dataset-name bug in the solution loader, where a multi-segment relative path on Windows leaked into and duplicated part of the Leontief cache filename. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Handle pandas `Series` and empty dataframes correctly in `get_full_ts()` and stop it from overwriting the index. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Fix results handling for solutions run with rolling horizon enabled. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Resolve documentation build warnings and errors, correcting broken cross-references, tutorial structure, heading formatting, and outdated dataset example paths. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Fix remaining mypy type errors across the codebase to allow stricter type checking going forward. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]

### Documentation Changes 📝
- Rewrite the mathematical formulation around the current constraint docstrings and updated notation, add unique labels to automatically numbered equations, fix cross-references, and document the zero-discount annuity case. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Restructure the tutorials into a linear progression, add results-API-reference, troubleshooting, and time-representation pages, move the scaling and scenario-tool guides into "ZEN-garden in detail", and remove the PWA (nonlinear CAPEX) documentation and superseded legacy tutorials. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Refresh and simplify the dataset documentation and dataset-creation tutorial, removing the obsolete piecewise-affine nonlinear-CAPEX example and adding missing parameter definitions to the example datasets. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Update installation, quick-start, and reference documentation to describe `uv`-based environment setup and the separately installable ZEN-temple visualization package. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Align API reference, module paths, and docstrings across the documentation with the refactored core module structure. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]

### Maintenance Tasks 🧹
- Reorganize the codebase into a more logical, element-based folder structure, grouping constraints, variables, and parameters by element type and introducing `workflow/` and restructured `utils/` subpackages. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Update the `ruff` and `mypy` tooling versions and add `pandas-stubs` and a `basedpyright` configuration for stricter development-time type checking. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Add `netCDF4` as a runtime dependency to support the new results storage format. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Migrate remaining JSON test fixtures (e.g., `test_1b`) to YAML and add missing default parameter values and units to test fixtures. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Expand and clean up test coverage, including a second technology in the diffusion-limit regression case, a fix for degeneracy in `test_1g`, a new `test_1j`, and removal of the obsolete `test_7b` dataset and the operation-only CLI module. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Remove unused code, including the obsolete `postprocess/results/cache.py` module and the stale piecewise-constraint wrapper. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Upgrade to TSAM v4 [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- move code from results class to compute energy balance dataframes to ZEN-temple. [[🔀 PR #1281](https://github.com/ZEN-universe/ZEN-garden/pull/1281) @manud99]

### BREAKING CHANGES ⚠️
- Store optimization results as NetCDF files. Results are now written and read as xarray datasets through `netCDF4` instead of the previous per-component format, which changes the output-folder layout and requires re-running models to obtain readable results. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Rewrite the solution loader and `Results` API. `Results.get_df()` is renamed to `get_unprocessed_result()`, `Results.solution_loader.scenarios` becomes `Results.scenarios`, the `get_total_per_scenario`/`get_full_ts_per_scenario` helpers are removed, and the remaining methods return all scenarios when no `scenario_name` is passed. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Refactor the model-construction core around a single `ModelSchema` source of truth. `EnergySystem` becomes a subclass of `Element`, `ElementRegistry`/`ElementFactory` are split, constraints are built in the element classes rather than per-element `ModelConstructor` objects, `create_custom_set` no longer takes a class, `config` is removed from `ServiceContainer`, and input validation moves from `InputDataChecks` into the config layer. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Convert all model input files from JSON to YAML. `config`, `system`, `scenario`, `attributes`, `base_units`, and `parameter_interpolation_off` files can now be written as YAML; the default config file becomes `config.yaml` (JSON now needs an explicit CLI flag), support for `base_units.csv` and `scenario.py` files is dropped, and comments become possible in input files. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Remove the operations wrapper, which is moved to a plugin, and drop support for the old `__main__.py` entry point in favor of `runner.py` and a new package entry point. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Remove the deprecated `time_steps` argument from `DataInput.extract_input_data`. Time-step classification is now handled automatically via `index_sets`. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Move the visualization CLI to the separately installable ZEN-temple package. The bundled `zen-visualization` entry point and `zen-temple` dependency are removed from ZEN-garden. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]
- Removed PWA for nonlinear capex approximation. [[🔀 PR #1326](https://github.com/ZEN-universe/ZEN-garden/pull/1326) @jacob-mannhardt]

## [v2.13.0] - 2026-07-01 

### New Features ✨
- Infer the time step type from the index sets in `extract_input_data()`, and remove `time_step` argument. Align the time step naming internally to `set_hours` and `set_years` to make them clearer [[🔀 PR #1282](https://github.com/ZEN-universe/ZEN-garden/pull/1282) @jacob-mannhardt]

## [v2.12.2] - 2026-06-17 

### Bug Fixes 🐛
- make code compatible with Pandas v3.0.0. [[🔀 PR #1278](https://github.com/ZEN-universe/ZEN-garden/pull/1278) @csfunke]

### Maintenance Tasks 🧹
- upgrade `black` to patch security vulnerability. [[🔀 PR #1276](https://github.com/ZEN-universe/ZEN-garden/pull/1276) @csfunke]
- Add a test for the comparisons module to `test_4a` [[🔀 PR #1269](https://github.com/ZEN-universe/ZEN-garden/pull/1269) @jacob-mannhardt]

## [v2.12.1] - 2026-06-03 

### Bug Fixes 🐛
- Fix a small bug in the comparison module that broke the routine. [[🔀 PR #1267](https://github.com/ZEN-universe/ZEN-garden/pull/1267) @jacob-mannhardt]

## [v2.12.0] - 2026-05-19 

### New Features ✨
- install `pre-commit` to run `ruff --fix` and `black` before creating new commits. [[🔀 PR #1258](https://github.com/ZEN-universe/ZEN-garden/pull/1258) @manud99]

### Documentation Changes 📝
- update links in plugin documentation [[🔀 PR #1254](https://github.com/ZEN-universe/ZEN-garden/pull/1254) @JeanWi]
- add plugin architecture to api reference [[🔀 PR #1254](https://github.com/ZEN-universe/ZEN-garden/pull/1254) @JeanWi]

## [v2.11.0] - 2026-04-10 

### New Features ✨
- implement a plugin architecture that allows easy extension of the core ZEN-garden code. [[🔀 PR #1252](https://github.com/ZEN-universe/ZEN-garden/pull/1252) @csfunke]

## [v2.10.0] - 2026-04-08 

### New Features ✨
- add option to export reduced costs to reduced_costs_dict.h5 [[🔀 PR #1231](https://github.com/ZEN-universe/ZEN-garden/pull/1231) @JeanWi]

### Documentation Changes 📝
- Fix wrong country index in tutorial example. The capacity_DE variable in docs/files/tutorial/01_analyze_outputs.rst was mistakenly using "CH" instead of "DE" as the index argument in r.get_total(...), which would have returned Swiss instead of German capacity. [[🔀 PR #1244](https://github.com/ZEN-universe/ZEN-garden/pull/1244) @felixduemig]

## [v2.9.8] - 2026-03-27 

### Bug Fixes 🐛
- remove lines causing TypeError in ZEN-temple [[🔀 PR #1233](https://github.com/ZEN-universe/ZEN-garden/pull/1233) @manud99]

### Documentation Changes 📝
- fix tuorial on adding technologies and carriers. The exercise questions previously did not match the solution. [[🔀 PR #1232](https://github.com/ZEN-universe/ZEN-garden/pull/1232) @jojoethz]
- update the contribution guide to include new rules on formatting and linting. [[🔀 PR #1234](https://github.com/ZEN-universe/ZEN-garden/pull/1234) @csfunke]
- create a new section in the developer guide on testing. [[🔀 PR #1234](https://github.com/ZEN-universe/ZEN-garden/pull/1234) @csfunke]
- add warnings and link to Python. New Python users find a link to Python.org. Also added warnings for the following issues: File paths that exceed 260 characters may lead to errors with Windows and special characters are not compatible with ZEN-garden. [[🔀 PR #1228](https://github.com/ZEN-universe/ZEN-garden/pull/1228) @johburger]

### Maintenance Tasks 🧹
- remove test dataset that was accidentally pushed to the ZEN-garden root directory. [[🔀 PR #1239](https://github.com/ZEN-universe/ZEN-garden/pull/1239) @csfunke]

## [v2.9.7] - 2026-02-10 

### Bug Fixes 🐛
- make the solver dir path relative to the config path, not the cwd. [[🔀 PR #1226](https://github.com/ZEN-universe/ZEN-garden/pull/1226) @jacob-mannhardt]

### Documentation Changes 📝
- Format docstrings so that equations and line breaks are consistent. Also replace `\mathrm{}` for text within equations with `\text{}` [[🔀 PR #1224](https://github.com/ZEN-universe/ZEN-garden/pull/1224) @johburger]

## [v2.9.6] - 2026-02-09 

### Bug Fixes 🐛
- Fixes bug when data folder is not in the cwd but in a different location. Allows the path to be different and saves the results there. [[🔀 PR #1222](https://github.com/ZEN-universe/ZEN-garden/pull/1222) @jacob-mannhardt]

### Maintenance Tasks 🧹
- reformat and lint ZEN-garden code to match style guidelines. The code now passes checks from the formatter ``Black`` and the linter ``Ruff``. [[🔀 PR #1220](https://github.com/ZEN-universe/ZEN-garden/pull/1220) @csfunke]
- enforce code formatting (via ``Black``) and linting (via ``Ruff``). All future pull requests must pass these checks to be eligible for merge into the ``main`` branch of ZEN-garden. These checks can be tested locally in a terminal by (i) activating the ZEN-garden environment, (ii) navigating the the ZEN-garden root folder, and (iii) typing ``black .`` and ``ruff --check .`` . [[🔀 PR #1220](https://github.com/ZEN-universe/ZEN-garden/pull/1220) @csfunke]

## [v2.9.5] - 2026-02-06 

### Bug Fixes 🐛
- fix result extraction from hdf files when there is a single column. [[🔀 PR #1216](https://github.com/ZEN-universe/ZEN-garden/pull/1216) @csfunke]

## [v2.9.4] - 2026-02-06 

### Bug Fixes 🐛
- fix version check from  `2.9.1` to `2.9.2` for new results. [[🔀 PR #1213](https://github.com/ZEN-universe/ZEN-garden/pull/1213) @jacob-mannhardt]

## [v2.9.3] - 2026-02-06 

### Bug Fixes 🐛
- add error catch for when extracting the units in the new format without updating the environment before. [[🔀 PR #1211](https://github.com/ZEN-universe/ZEN-garden/pull/1211) @jacob-mannhardt]

## [v2.9.2] - 2026-02-06 

### Bug Fixes 🐛
- Make result reading faster by splitting the `value` and `unit` columns into two keys in the `.h5` file. [[🔀 PR #1209](https://github.com/ZEN-universe/ZEN-garden/pull/1209) @jacob-mannhardt]

## [v2.9.1] - 2026-02-05 

### Bug Fixes 🐛
- Set `macos` version to `latest` instead of `macos13` because deprecated [[🔀 PR #1207](https://github.com/ZEN-universe/ZEN-garden/pull/1207) @jacob-mannhardt]
- Skip `read_components` when the scenario does not exist [[🔀 PR #1207](https://github.com/ZEN-universe/ZEN-garden/pull/1207) @jacob-mannhardt]
- Create `ureg` (including reading in the user units) only for one scenario, not for all [[🔀 PR #1207](https://github.com/ZEN-universe/ZEN-garden/pull/1207) @jacob-mannhardt]
- Move the components construction outside the initialization of the scenarios. The components are only created upon requests when the data is actually read. [[🔀 PR #1207](https://github.com/ZEN-universe/ZEN-garden/pull/1207) @jacob-mannhardt]

## [v2.9.0] - 2026-01-22 

### New Features ✨
- implement ``zen-operation`` wrapper. This wrapper allows users to seamlessly run operation-only scenarios using the capacity values of a previous simulation. Users may provide a new ``scenarios_op`` file that specifies the operational scenarios to run. This new feature replaces the old configuration ``include_operation_only_phase``, which has now been removed. [[🔀 PR #1204](https://github.com/ZEN-universe/ZEN-garden/pull/1204) @csfunke]

### Documentation Changes 📝
- implement detailed Google-style docstrings for the ``UnitHandling`` class. [[🔀 PR #1204](https://github.com/ZEN-universe/ZEN-garden/pull/1204) @csfunke]
- improve tutorial of operation-only simulations and update the tutorial to include the new ``zen-operation`` wrapper. [[🔀 PR #1204](https://github.com/ZEN-universe/ZEN-garden/pull/1204) @csfunke]

### Maintenance Tasks 🧹
- create test cases for the new ``zen-operation`` wrapper. [[🔀 PR #1204](https://github.com/ZEN-universe/ZEN-garden/pull/1204) @csfunke]
- suppress ``Pint`` package output on redefining units. This output was previously printed to the terminal whenever a new ``Results`` object was initialized. [[🔀 PR #1204](https://github.com/ZEN-universe/ZEN-garden/pull/1204) @csfunke]

## [v2.8.13] - 2026-01-19 

### Bug Fixes 🐛
- return empty series when there are no series to concatenate in `_combine_dataseries` in `solution_loader.py`. [[🔀 PR #1201](https://github.com/ZEN-universe/ZEN-garden/pull/1201) @manud99]

### Documentation Changes 📝
- fix broken links in the README file. Some of the documentation links were outdated and not longer worked. [[🔀 PR #1196](https://github.com/ZEN-universe/ZEN-garden/pull/1196) @csfunke]

### Maintenance Tasks 🧹
- add continuous integration workflow that checks code formatting, linting, and type checking. Uses the packages `black` for formatting, `ruff` for linting, and `mypy` for type checking. Errors are reported but not enforced initially, allowing developers time to clean up the existing codebase. Enforcement will be enabled once all errors are resolved. [[🔀 PR #1199](https://github.com/ZEN-universe/ZEN-garden/pull/1199) @csfunke]

## [v2.8.12] - 2026-01-14 

### Bug Fixes 🐛
- fix overwriting the values for a specific year when only one year is selected [[🔀 PR #1193](https://github.com/ZEN-universe/ZEN-garden/pull/1193) @jacob-mannhardt]

### Maintenance Tasks 🧹
- delete ``.bumpversion.cfg`` file. It is now obsolete, as version bumping is performed via a custom Python script rather than the bump2version package. [[🔀 PR #1190](https://github.com/ZEN-universe/ZEN-garden/pull/1190) @csfunke]
- correct spelling in changelog. In previous changelog versions, the header "Maintenance Tasks" was spelled wrong. [[🔀 PR #1190](https://github.com/ZEN-universe/ZEN-garden/pull/1190) @csfunke]
- skip release in CI workflow when no version bump occurs. [[🔀 PR #1188](https://github.com/ZEN-universe/ZEN-garden/pull/1188) @csfunke]

## [v2.8.11] - 2026-01-06 

### Bug Fixes 🐛
- fix bug when extracting `get_full_ts("storage_level",year=2022)` for a solution with rolling horizon. [[🔀 PR #1186](https://github.com/ZEN-universe/ZEN-garden/pull/1186) @jacob-mannhardt]

### Documentation Changes 📝
- clean changelog. [[🔀 PR #1184](https://github.com/ZEN-universe/ZEN-garden/pull/1184) @csfunke]

## [v2.8.10] - 2026-01-05 

### Bug Fixes 🐛
- fix PyPi release by adding env variable in `create_tag` [[🔀 PR #1182](https://github.com/ZEN-universe/ZEN-garden/pull/1182) @jacob-mannhardt]

## [v2.8.9] - 2026-01-05 

### Bug Fixes 🐛
- fix PyPi release by making the NEW_VERSION string accessible across jobs [[🔀 PR #1180](https://github.com/ZEN-universe/ZEN-garden/pull/1180) @jacob-mannhardt]

## [v2.8.8] - 2026-01-05 

### Maintenance Tasks 🧹
- test release to PyPi [[🔀 PR #1177](https://github.com/ZEN-universe/ZEN-garden/pull/1177) @jacob-mannhardt]

## [v2.8.7] - 2026-01-05 

### Bug Fixes 🐛
- fix bug in or myopic foresight results when yearly series is empty (index returns empty results). Returns empty Series. [[🔀 PR #1172](https://github.com/ZEN-universe/ZEN-garden/pull/1172) @jacob-mannhardt]

### Documentation Changes 📝
- add changelog to documentation. The changelog now gets copied to the ``docs/files/api/generated`` folder when the documentation is built. This allows it to be shown in the "References" section of the documentation. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]

### Maintenance Tasks 🧹
- fix branch deletion in continuous integration pipeline. The previous pipeline attempted to delete a branch which is no longer in use. [[🔀 PR #1173](https://github.com/ZEN-universe/ZEN-garden/pull/1173) @csfunke]
- update pull request template to match changelog automation. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]
- implement semantic version bumping. Major version bumps are now triggered upon breaking changes; minor version bumps are triggered by new features; patch version bumps are triggered by bug fixes. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]
- automate change log. Information for the change log is now taken from the ``Detailed list of changes`` section of the pull request body. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]

## [v0.1.0] - [v2.8.4]

No release notes exist for ZEN garden versions 0.1.0 -> 2.8.4.