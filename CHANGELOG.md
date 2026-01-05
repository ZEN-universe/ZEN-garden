# Changelog

This file gets automatically updated in ZEN-garden's continuous integration 
procedures. Do not edit the file manually.

## [v2.8.7] - 2026-01-05 

### Bug Fixes 🐛
- bug fix for myopic foresight results when yearly series is empty (index returns empty results). Returns empty Series. [[🔀 PR #1172](https://github.com/ZEN-universe/ZEN-garden/pull/1172) @jacob-mannhardt]

### Documentation Changes 📝
- add changelog to documentation. The changelog now gets copied to the ``docs/files/api/generated`` folder when the documentation is built. This allows it to be shown in the "References" section of the documentation. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]

### Maintainance Tasks 🧹
- update pull request template to match changelog automation. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]
- implement semantic version bumping. Major version bumps are now triggered upon breaking changes; minor version bumps are triggered by new features; patch version bumps are triggered by bug fixes. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]
- automate change log. Information for the change log is now taken from the ``Detailed list of changes`` section of the pull request body. [[🔀 PR #1169](https://github.com/ZEN-universe/ZEN-garden/pull/1169) @csfunke]

## [v4.0.0] - 2026-01-05 

### New Features ✨
- describe new features added to the model. Features include any new functionality that is available to ZEN-garden users. New features automatically lead to minor version bumps. [[🔀 PR #1167](https://github.com/ZEN-universe/ZEN-garden/pull/1167) @csfunke]

### Bug Fixes 🐛
- describe bug fixed through the pull-request, including 1-2 additional sentances on the context. Bug fixes automatically lead to patch version bumps. [[🔀 PR #1167](https://github.com/ZEN-universe/ZEN-garden/pull/1167) @csfunke]

### Documentation Changes 📝
- describe changes to the documentation. This category is for all changes to the documentation or docstrings. Documentation changes do not bump the ZEN-garden version. [[🔀 PR #1167](https://github.com/ZEN-universe/ZEN-garden/pull/1167) @csfunke]
- add class diagrams to documentationn. This category is for all changes to the documentation or to docstrings. Documentation changes will not bump the ZEN-garden version. [[🔀 PR #1165](https://github.com/ZEN-universe/ZEN-garden/pull/1165) @csfunke]

### Maintainance Tasks 🧹
- describe maintainance tasks such as updating tests, improving continuous integration workflows, and refactoring code. These tasks do not change the functionality of ZEN-garden from a user perspective and therefore do not lead to a version bump. They are primarily relevant for developers. [[🔀 PR #1167](https://github.com/ZEN-universe/ZEN-garden/pull/1167) @csfunke]
- refactor default_config. Chores include any maintainance tasks such updating tests, improving continuous integration workflows, and refactoring code. They do not change the functionality of ZEN-garden from a user perspective and therefore do not lead to a version bump. [[🔀 PR #1165](https://github.com/ZEN-universe/ZEN-garden/pull/1165) @csfunke]

### BREAKING CHANGES ⚠️
- describe breaking changes. Add a 1–2 sentence description of the breaking change. Breaking changes automatically lead to a major version bump. [[🔀 PR #1167](https://github.com/ZEN-universe/ZEN-garden/pull/1167) @csfunke]

## [v0.1.0] - [v2.8.4]

No release notes exist for ZEN garden versions 0.1.0 -> 2.8.4 are unavailable.