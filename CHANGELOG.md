# Changelog

All notable changes to this project will be documented in this file.

## [v1.2.0] - 2024-05-08
### Added
- new dataset creation tutorials and updated `dataset_creation_tutorial.md`

### Fixed
- IIS calculation and printing
- Fix technology diffusion constraints 
  
### Changed
- Upgraded to `linopy>=0.3.8`
- Vectorized all continuous constraints
- ´carbon_intensity_carrier´ is split in´carbon_intensity_carrier_import´ and ´carbon_intensity_carrier_export´
- attributes of retrofitting technologies are combined in one
- test 3a and 3b are replaced by test 1e
- test 5a and 5b were combined 

### Removed
- `return_constraints` statement for all continuous constraints
- rule-based and block-based constraint distinction
- `breakpoints_pwa_capex.csv` file no longer used
  
## [v1.1.0] - 2024-02-28
### Added
- Exponential transport loss factor
  - added test 
- Saving units of parameters and variables and accessing them in results

### Fixed
- Docstring reading in results
- Minor bug fixes
  
### Changed
- Result handling
- Parameter names:
  - `set_carriers`:
    - `carbon_intensity` to `carbon_intensity_carrier`
  - `set_technologies`:
    - `carbon_intensity` to `carbon_intensity_technology`
  - `set_storage_technologies`:
    - `capex_specific` to `capex_specific_storage`
    - `capex_specific_energy` to `capex_specific_storage_energy`
  - `set_transport_technologies`:
    - `transport_loss_factor` to `transport_loss_factor_linear`
- Move `use_capacities_existing` to `system`
      
### Removed
- "linopy" name in environment
- Solver options from default config
  
## [v1.0.3] - 2024-02-14 ❤️
### Added
- Flexible `results.py` structure with abstract solution loader
- `comparison` module

### Removed
- standard plots
  
## [v1.0.2] - 2024-01-29
### Added
- compare parameters with different shapes in `r.compare_model_parameters()`
  
### Fixed
- Writing IIS to file
- set expansion of `set_technologies` in scenario routine

### Changed
- convert `how_to_zen-garden.pdf` and `dataset_creation_tutorial.pdf` into `.md`

## [v1.0.1] - 2024-01-02
### Added
- Unit consistency checks: Check that all units are internally consistent across parameters
  
### Fixed
- Bug fix of numerical values regarding construction time:
1. Sometimes forced the capacity additions in the last and the second last time step to be equal
2. construction time was one period too short
- Bug fix of the lifetime of existing capacities (existed one period longer than desired)
- Adapted the tests accordingly

### Changed
- Simplified calculation of lifetime and construction time (Issue [#257])
- Sped up parameter and constraint construction by removing time step encoder-decoder (Issue [#362])

## [v1.0.0] - 2023-12-11

Beginning of versioning.

### Added
- Internal calculation of haversine distance (Issue [#310])
- Retrofitting

### Fixed
- Fix unit handling for singular dimensionality matrices
- Smaller fixes
  
### Changed
- `attributes.json` instead of `attributes.csv` (Issue [#339])
- `energy_system` instead of `system_specification`

### Removed
- PWA of conversion factor (Issue [#343])
- Technology-dependent time steps (Issue [#290])
- Don't show plots in tests anymore
