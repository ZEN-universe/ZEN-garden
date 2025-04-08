Closes # (if applicable).

## Changes proposed in this Pull Request


## Checklist
Please check all items that apply. If an item is not applicable, please remove it from the list.

### PR structure
- [ ] The PR has a descriptive title.
- [ ] The corresponding issue is linked with # in the PR description.

### Code quality
- [ ] Newly introduced dependencies are added to `pyproject.toml`.
- [ ] Tests for new features were added:
  - [ ] A new test folder is added or an existing test folder is adapted
  - [ ] The test is added to `run_tests.py` and `docu_test_cases.md`
  - [ ] The tested variables are added to `test_variables.json`
  - [ ] Code changes have been tested locally and all tests pass

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
