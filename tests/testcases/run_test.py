"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      September-2022
Authors:      Janis Fluri (janis.fluri@id.ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""
from zen_garden._internal import main

import pytest
from copy import deepcopy
import pandas as pd
import os

# fixture to get the default config (always as new instance)
@pytest.fixture
def config():
    from config import config
    return deepcopy(config)

def compare_variables(test_model, optimization_setup,folder_path):
    # import csv file containing selected variable values of test model collection
    test_variables = pd.read_csv(os.path.join(folder_path, 'test_variables_readable.csv'),header=0, index_col=None)
    # dictionary to store variable names, indices, values and test values of variables which don't match the test values
    failed_variables = {}
    # iterate through dataframe rows
    for data_row in test_variables.values:
        # skip row if data doesn't correspond to selected test model
        if data_row[0] != test_model:
            continue
        # get variable attribute of optimization_setup object by using string of the variable's name (e.g. optimization_setup.model.importCarrierFLow)
        variable_attribute = getattr(optimization_setup.model,data_row[1])
        # iterate through indices of current variable
        for variable_index in variable_attribute.extract_values():
            # ensure equality of dataRow index and variable index
            if str(variable_index) == data_row[2]:
                # check if variable value at current index differs from zero such that relative error can be computed
                if variable_attribute.extract_values()[variable_index] != 0:
                    # check if relative error exceeds limit of 10^-3, i.e. value differs from test value
                    if abs(variable_attribute.extract_values()[variable_index] - data_row[3]) / variable_attribute.extract_values()[variable_index] > 10**(-3):
                        if data_row[1] in failed_variables:
                            failed_variables[data_row[1]][data_row[2]] = {"computedValue" : variable_attribute.extract_values()[variable_index]}
                        else:
                            failed_variables[data_row[1]] = {data_row[2] : {"computedValue" : variable_attribute.extract_values()[variable_index]}}
                        failed_variables[data_row[1]][data_row[2]]["test_value"] = data_row[3]
                else:
                    # check if absolute error exceeds specified limit
                    if abs(variable_attribute.extract_values()[variable_index] - data_row[3]) > 10**(-3):
                        if data_row[1] in failed_variables:
                            failed_variables[data_row[1]][data_row[2]] = {"computedValue" : variable_attribute.extract_values()[variable_index]}
                        else:
                            failed_variables[data_row[1]] = {data_row[2] : {"computedValue" : variable_attribute.extract_values()[variable_index]}}
                        failed_variables[data_row[1]][data_row[2]]["test_value"] = data_row[3]
    assertion_string = str()
    for failed_var in failed_variables:
        assertion_string += f"\n{failed_var}{failed_variables[failed_var]}"

    return failed_variables, assertion_string


# All the tests
###############

def test_1a(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_1a"))

    failed_variables, assertion_string = compare_variables("test_1a",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_1b(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_1b"))

    failed_variables, assertion_string = compare_variables("test_1b",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_1c(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_1c"))

    failed_variables, assertion_string = compare_variables("test_1c",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_1d(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_1d"))

    failed_variables, assertion_string = compare_variables("test_1d",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_2a(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_2a"))

    failed_variables, assertion_string = compare_variables("test_2a",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_2b(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_2b"))

    failed_variables, assertion_string = compare_variables("test_2b",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_2c(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_2c"))

    failed_variables, assertion_string = compare_variables("test_2c",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_2d(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_2d"))

    failed_variables, assertion_string = compare_variables("test_2d",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_3a(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_3a"))

    failed_variables, assertion_string = compare_variables("test_3a",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_3b(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_3b"))

    failed_variables, assertion_string = compare_variables("test_3b",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_4a(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_4a"))

    failed_variables, assertion_string = compare_variables("test_4a",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_4b(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_4b"))

    failed_variables, assertion_string = compare_variables("test_4b",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_4c(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_4c"))

    failed_variables, assertion_string = compare_variables("test_4c",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_4d(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_4d"))

    failed_variables, assertion_string = compare_variables("test_4d",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_4e(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_4e"))

    failed_variables, assertion_string = compare_variables("test_4e",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_4f(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_4f"))

    failed_variables, assertion_string = compare_variables("test_4f",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_4g(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_4g"))

    failed_variables, assertion_string = compare_variables("test_4g",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_5a(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_5a"))

    failed_variables, assertion_string = compare_variables("test_5a",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_6a(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_6a"))

    failed_variables, assertion_string = compare_variables("test_6a",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"

def test_7a(config):
    # run the test
    folder_path = os.path.dirname(__file__)
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, "test_7a"))

    failed_variables, assertion_string = compare_variables("test_7a",optimization_setup,folder_path)
    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"