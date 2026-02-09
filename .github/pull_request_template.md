## Summary

Provide a brief summary of the changes proposed in this pull request.

Closes # (if applicable).


## Detailed list of changes

List all changes proposed in the pull request in the format `<type>: <description>` (**mandatory**). This list will be used to update the changelog. Valid types include `fix`, `feat`, `docs`, `chore`, and `breaking`.

The first sentence of the description should be written in the imperative tense (e.g., "Add new feature" or "Clean existing code file"). Subsequent sentences may have any format; however, the description must consist of only one paragraph (no newline characters).

An example list is shown below. Update these sections to match the changes proposed in the pull request.:

- fix: describe bug fixed through the pull-request, including 1-2 additional sentences on the context. Bug fixes automatically lead to patch version bumps.
- feat: describe new features added to the model. Features include any new functionality that is available to ZEN-garden users. New features automatically lead to minor version bumps.
- docs: describe changes to the documentation. This category is for all changes to the documentation or docstrings. Documentation changes do not bump the ZEN-garden version.
- chore: describe maintenance tasks such as updating tests, improving continuous integration workflows, and refactoring code. These tasks do not change the functionality of ZEN-garden from a user perspective and therefore do not lead to a version bump. They are primarily relevant for developers.
- breaking: describe breaking changes. Add a 1–2 sentence description of the breaking change. Breaking changes automatically lead to a major version bump.


## Checklist

Please check all items that apply. If an item is not applicable, please remove it from the list.

### PR structure
- [ ] The PR has a descriptive title.
- [ ] The corresponding issue is linked with `#` in the PR description.
- [ ] A detailed list of changes is provided.


### Code quality
- [ ] Newly introduced dependencies are added to `pyproject.toml`.
- [ ] Code changes have been tested locally and all tests pass.
- [ ] Code has been formatted via ``black .`` in a terminal window.
- [ ] Linter ``ruff --check .`` passes all checks.
- [ ] Tests for new features were added:
  - [ ] The test is added to `run_tests.py` and `docu_test_cases.md`.
  - [ ] The tested variables are added to `test_variables.json`.


### Code changes
- [ ] If the name of an existing parameter is changed, both the new and old names are added to `zen_garden/preprocess/parameter_change_log.py`.
- [ ] If a new parameter is added, the default value (0, 1, or `np.inf` allowed) and a parameter with the same unit are added to `zen_garden/preprocess/parameter_change_log.py`.
- [ ] If the name of an existing variable is changed and the variable is used in the visualization platform, the name change is added to `variable_versions` in the [ZEN-temple code](https://github.com/ZEN-universe/ZEN-temple/blob/main/src/zen_temple/utils.py).


### Documentation
- [ ] Code changes are sufficiently documented (e.g., new functions contain docstrings).
- [ ] Changes to parameters, variables, sets, or constraints are added to `docs/files/zen_garden_in_detail/sets_params_constraints.rst` and `docs/files/zen_garden_in_detail/mathematical_formulation.rst`.
- [ ] Changes to the configuration are added to `docs/files/zen_garden_in_detail/configurations.rst`.
- [ ] Additional features are added to `docs/files/zen_garden_in_detail/additional_features.rst`.
- [ ] Other changes are documented in the corresponding section of the documentation.
