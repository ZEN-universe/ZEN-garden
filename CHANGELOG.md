# Changelog

This file gets automatically updated in ZEN-garden's continuous integration 
procedures. Do not edit the file manually.

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

No release notes exist for ZEN garden versions 0.1.0 -> 2.8.4 are unavailable.