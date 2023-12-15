# Changelog

All notable changes to this project will be documented in this file.

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
