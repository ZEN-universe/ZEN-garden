# Changelog

This file gets automatically updated in ZEN-garden's continuous integration 
procedures. Do not edit the file manually.

## [v2.8.11] - 2026-01-06 

### Bug Fixes 🐛
- bug fix when extracting `get_full_ts("storage_level",year=2022)` for a solution with rolling horizon. [[🔀 PR #1186](https://github.com/ZEN-universe/ZEN-garden/pull/1186) @jacob-mannhardt]

### Documentation Changes 📝
- clean changelog. [[🔀 PR #1184](https://github.com/ZEN-universe/ZEN-garden/pull/1184) @csfunke]

## [v2.8.10] - 2026-01-05 

### Bug Fixes 🐛
- fix PyPi release by adding env variable in `create_tag` [[🔀 PR #1182](https://github.com/ZEN-universe/ZEN-garden/pull/1182) @jacob-mannhardt]

## [v2.8.9] - 2026-01-05 

### Bug Fixes 🐛
- fix PyPi release by making the NEW_VERSION string accessible across jobs [[🔀 PR #1180](https://github.com/ZEN-universe/ZEN-garden/pull/1180) @jacob-mannhardt]

## [v2.8.8] - 2026-01-05 

### Maintainance Tasks 🧹
- test release to PyPi [[🔀 PR #1177](https://github.com/ZEN-universe/ZEN-garden/pull/1177) @jacob-mannhardt]

## [v2.8.7] - 2026-01-05 

### Bug Fixes 🐛
- fix bug in or myopic foresight results when yearly series is empty (index returns empty results). Returns empty Series. [[🔀 PR #1172](https://github.com/ZEN-universe/ZEN-garden/pull/1172) @jacob-mannhardt]

### Documentation Changes 📝
- add changelog to documentation. The changelog now gets copied to the ``docs/files/api/generated`` folder when the documentation is built. This allows it to be shown in the "References" section of the documentation. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]

### Maintainance Tasks 🧹
- fix branch deletion in continuous integration pipeline. The previous pipeline attempted to delete a branch which is no longer in use. [[🔀 PR #1173](https://github.com/ZEN-universe/ZEN-garden/pull/1173) @csfunke]
- update pull request template to match changelog automation. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]
- implement semantic version bumping. Major version bumps are now triggered upon breaking changes; minor version bumps are triggered by new features; patch version bumps are triggered by bug fixes. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]
- automate change log. Information for the change log is now taken from the ``Detailed list of changes`` section of the pull request body. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]

## [v0.1.0] - [v2.8.4]

No release notes exist for ZEN garden versions 0.1.0 -> 2.8.4 are unavailable.