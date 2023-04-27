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
import numpy as np

from zen_garden._internal import main
from zen_garden.postprocess.results import Results

import pytest
from copy import deepcopy
import pandas as pd
import os
from collections import defaultdict

# fixtures
##########

@pytest.fixture
def config():
    """
    :return: A new instance of the config
    """
    from config import config
    return deepcopy(config)

@pytest.fixture
def folder_path():
    """
    :return: Returns the path of the testcase folder
    """
    return os.path.dirname(__file__)


# helper functions
##################

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
                            failed_variables[data_row[1]][data_row[2]] = {"computed_values" : variable_attribute.extract_values()[variable_index]}
                        else:
                            failed_variables[data_row[1]] = {data_row[2] : {"computed_values" : variable_attribute.extract_values()[variable_index]}}
                        failed_variables[data_row[1]][data_row[2]]["test_value"] = data_row[3]
                else:
                    # check if absolute error exceeds specified limit
                    if abs(variable_attribute.extract_values()[variable_index] - data_row[3]) > 10**(-3):
                        if data_row[1] in failed_variables:
                            failed_variables[data_row[1]][data_row[2]] = {"computed_values" : variable_attribute.extract_values()[variable_index]}
                        else:
                            failed_variables[data_row[1]] = {data_row[2] : {"computed_values" : variable_attribute.extract_values()[variable_index]}}
                        failed_variables[data_row[1]][data_row[2]]["test_value"] = data_row[3]
    assertion_string = str()
    for failed_var in failed_variables:
        assertion_string += f"\n{failed_var}{failed_variables[failed_var]}"

    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"


def compare_variables_results(test_model: str, results: Results, folder_path: str):
    """
    Compares the variables of a Results object from the test run to precomputed values
    :param test_model: The model to test (name of the data set)
    :param results: The Results object
    :param folder_path: The path to the folder containing the file with the correct variables
    """
    # import csv file containing selected variable values of test model collection
    test_variables = pd.read_csv(os.path.join(folder_path, 'test_variables_readable.csv'),header=0, index_col=None)
    # dictionary to store variable names, indices, values and test values of variables which don't match the test values
    failed_variables = defaultdict(dict)
    # iterate through dataframe rows
    for data_row in test_variables[test_variables["test"] == test_model].values:
        # get the corresponding data frame from the results
        variable_df = results.get_df(data_row[1])
        # iterate through indices of current variable
        for variable_index, variable_value in variable_df.items():
            # ensure equality of dataRow index and variable index
            if str(variable_index) == data_row[2]:
                # check if close
                if not np.isclose(variable_value, data_row[3], rtol=1e-3):
                    failed_variables[data_row[1]][data_row[2]] = {"computed_values": variable_value,
                                                                  "test_value": data_row[3]}
    # create the string of all failed variables
    assertion_string = ""
    for failed_var, failed_value in failed_variables.items():
        assertion_string += f"\n{failed_var}: {failed_value}"

    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"


# All the tests
###############

def test_1a(config, folder_path):
    # run the test
    data_set_name = "test_1a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1b(config, folder_path):
    # run the test
    data_set_name = "test_1b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1c(config, folder_path):
    # run the test
    data_set_name = "test_1c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1d(config, folder_path):
    # run the test
    data_set_name = "test_1d"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2a(config, folder_path):
    # run the test
    data_set_name = "test_2a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2b(config, folder_path):
    # run the test
    data_set_name = "test_2b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2c(config, folder_path):
    # run the test
    data_set_name = "test_2c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2d(config, folder_path):
    # run the test
    data_set_name = "test_2d"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3a(config, folder_path):
    # run the test
    data_set_name = "test_3a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3b(config, folder_path):
    # run the test
    data_set_name = "test_3b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4a(config, folder_path):
    # run the test
    data_set_name = "test_4a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4b(config, folder_path):
    # run the test
    data_set_name = "test_4b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4c(config, folder_path):
    # run the test
    data_set_name = "test_4c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4d(config, folder_path):
    # run the test
    data_set_name = "test_4d"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4e(config, folder_path):
    # run the test
    data_set_name = "test_4e"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4f(config, folder_path):
    # run the test
    data_set_name = "test_4f"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4g(config, folder_path):
    # run the test
    data_set_name = "test_4g"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_5a(config, folder_path):
    # run the test
    data_set_name = "test_5a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_6a(config, folder_path):
    # run the test
    data_set_name = "test_6a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_7a(config, folder_path):
    # run the test
    data_set_name = "test_7a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
