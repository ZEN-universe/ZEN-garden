## Summary

Provde a brief summary of the changes proposed in this pull request. 

Closes # (if applicable).


## Detailed list of changes

List all changes suggested in the pull request in the the format ``<type>:<description>`` (Mandatory!). This list will used to update the changelog. Valid types include ``fix``, ``feat``, ``docs``, ``chore``, and ``breaking``. The first sentance of the description should be in the imperative tense, i.e. "Add new feature" or "Clean existing code file". Subsequent sentances may have any format, however, the discription must consist of only one paragraph (no newline characters). Example list:

- fix: remove bug in changelog updates. Add a 1-2 sentance description of the bug. Bug fixes automatically lead to patch version bumps.
- feat: add piecewise affince capex representation. Add a 1-2 sentance description of the new feature. New features automatically lead to minor version bumps.
- docs: add class diagrams to documentationn. This category is for all changes to the documentation or to docstrings. Documentation changes will not bump the ZEN-garden version.
- chore: refactor default_config. Chores include any maintainance tasks such updating tests, improving continuous integration workflows, and refactoring code. They do not change the functionality of ZEN-garden from a user perspective and therefore do not lead to a version bump.
- breaking: remove run_module function (depricated). Add a 1-2 sentance description of the breaking change. Breaking changes automatically lead to a major version bump.

## Checklist
Please check all items that apply. If an item is not applicable, please remove 
it from the list.

### PR structure
- [ ] The PR has a descriptive title.
- [ ] The corresponding issue is linked with # in the PR description.
- [ ] Detailed list of changes was provided 


### Code quality
- [ ] Newly introduced dependencies are added to `pyproject.toml`.
- [ ] Code changes have been tested locally and all tests pass
- [ ] Tests for new features were added:
  - [ ] The test is added to `run_tests.py` and `docu_test_cases.md`
  - [ ] The tested variables are added to `test_variables.json`


### Code changes
- [ ] If the name of an existing parameter is changed, the new and old name are added to `zen_garden/preprocess/parameter_change_log.py`
- [ ] If a new parameter is added, add the default value (0, 1, or `np.inf` allowed) and a parameter with the same unit are added to `zen_garden/preprocess/parameter_change_log.py`
- [ ] If the name of an existing variable is changed and the variable is used in the visualization platform, the name change is added to `variable_versions` in the [ZEN-temple code](https://github.com/ZEN-universe/ZEN-temple/blob/main/src/zen_temple/utils.py)

### Documentation
- [ ] Code changes are sufficiently documented, i.e. new functions contain docstrings
- [ ] Changes in the parameters, variables, sets, or constraints are added to `docs/files/zen_garden_in_detail/sets_params_constraints.rst` and `docs/files/zen_garden_in_detail/mathematical_formulation.rst`
- [ ] Changes in the config are added to `docs/files/zen_garden_in_detail/configurations.rst`
- [ ] Additional features are added to `docs/files/zen_garden_in_detail/additional_features.rst`
- [ ] Other changes are documented in the corresponding section of the documentation
