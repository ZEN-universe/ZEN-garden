import json
import os
import re
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from zen_garden._internal import main
from zen_garden.postprocess.results import Results


# fixtures
##########


@pytest.fixture
def config():
    """
    :return: A new instance of the config
    """
    # TODO make work with new json! maybe use run_module from __main__.py directly
    from config import config

    config.solver.keep_files = False
    return deepcopy(config)


@pytest.fixture
def folder_path():
    """
    :return: Returns the path of the testcase folder
    """
    return os.path.dirname(__file__)


# helper functions
##################

def compare_variables_results(test_model: str, results: Results, folder_path: str):
    """
    Compares the variables of a Results object from the test run to precomputed values
    :param test_model: The model to test (name of the data set)
    :param results: The Results object
    :param folder_path: The path to the folder containing the file with the correct variables
    """
    # import json file containing selected variable values of test model collection
    with open(os.path.join(folder_path, "test_variables.json")) as f:
        test_variables = json.load(f)
    # dictionary to store variable names, indices, values and test values of variables which don't match the test values
    failed_variables = defaultdict(dict)
    compare_counter = 0
    # iterate through dataframe rows
    if test_model in test_variables:
        for s in test_variables[test_model]:
            if s in results.solution_loader.scenarios:
                test_values = test_variables[test_model][s]
                for c in test_values:
                    if c in results.solution_loader.components:
                        values = results.get_df(c)[s]
                        for test_value in test_values[c]:
                            if isinstance(test_value["index"],list):
                                test_index = tuple(test_value["index"])
                            else:
                                test_index = test_value["index"]
                            if test_index in values.index:
                                if not np.isclose(values[test_index], test_value["value"], rtol=1e-3):
                                    failed_variables[c][test_index] = {
                                        "computed_values": values[test_index],
                                        "test_value": test_value["value"],
                                    }
                                compare_counter += 1
                            else:
                                print(f"Index {test_value['index']} not found in results for component {c}")
                    else:
                        print(f"Component {c} not found in results")
            else:
                print(f"Scenario {s} not found in results")
    # create the string of all failed variables
    assertion_string = ""
    for failed_var, failed_value in failed_variables.items():
        assertion_string += f"\n{failed_var}: {failed_value}"

    assert (
        len(failed_variables) == 0
    ), f"The variables {assertion_string} don't match their test values"
    if compare_counter == 0:
        warnings.warn(UserWarning(f"No variables have been compared in {test_model}. If not intended, check the test_variables.json file."))


def check_get_total_get_full_ts(
    results: Results,
    specific_scenario=False,
    element_name=None,
    year=None,
    discount_to_first_step=True,
    get_doc=False,
):
    """
    Tests the functionality of the Results methods get_total() and get_full_ts()

    :param get_doc:
    :param discount_to_first_step: Apply annuity to first year of interval or entire interval
    :param element_name: Specific element
    :param year: Specific year
    :param specific_scenario: Specific scenario
    :param results: Results instance of testcase function has been called from
    """
    test_variables = ["demand", "capacity", "storage_level", "capacity_limit"]
    scenario = None
    if specific_scenario:
        scenario = next(iter(results.solution_loader.scenarios.keys()))
    for test_variable in test_variables:
        df_total = results.get_total(test_variable, scenario_name=scenario, year=year)
        if test_variable != "capacity_limit":
            df_full_ts = results.get_full_ts(
                test_variable,
                scenario_name=scenario,
                year=year,
                discount_to_first_step=discount_to_first_step,
            )
        if element_name is not None:
            df_total = results.get_total(
                test_variable, element_name=df_total.index[0][0]
            )
            if test_variable != "capacity_limit":
                df_full_ts = results.get_full_ts(
                    test_variable, element_name=df_full_ts.index[0][0]
                )
    if get_doc:
        results.get_doc(test_variables[0])


# All the tests
###############

def test_1a(config, folder_path):
    # add duals for this test
    config.solver.save_duals = True

    # run the test
    data_set_name = "test_1a"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_1b(config, folder_path):
    # run the test
    data_set_name = "test_1b"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1c(config, folder_path):
    # run the test
    data_set_name = "test_1c"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1d(config, folder_path):
    # run the test
    data_set_name = "test_1d"
    main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1e(config, folder_path):
    # run the test
    data_set_name = "test_1e"
    main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1f(config, folder_path):
    # run the test
    data_set_name = "test_1f"
    main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2a(config, folder_path):
    # run the test
    data_set_name = "test_2a"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2b(config, folder_path):
    # run the test
    data_set_name = "test_2b"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2c(config, folder_path):
    # run the test
    data_set_name = "test_2c"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3a(config, folder_path):
    # run the test
    data_set_name = "test_3a"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3b(config, folder_path):
    # run the test
    data_set_name = "test_3b"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_3c(config, folder_path):
    # run the test
    data_set_name = "test_3c"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res, year=0)


def test_3d(config, folder_path):
    # run the test
    data_set_name = "test_3d"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # compare the variables of the optimization setup ## disabled for myopic foresight tests!
    # compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res, discount_to_first_step=False)


def test_3e(config, folder_path):
    # run the test
    data_set_name = "test_3e"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # compare the variables of the optimization setup ## disabled for myopic foresight tests!
    # compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3f(config, folder_path):
    # run the test
    data_set_name = "test_3f"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # compare the variables of the optimization setup ## disabled for myopic foresight tests!
    # compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3g(config, folder_path):
    # run the test
    data_set_name = "test_3g"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_3h(config, folder_path):
    # run the test
    data_set_name = "test_3h"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_3i(config, folder_path):
    # run the test
    data_set_name = "test_3i"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_4a(config, folder_path):
    # run the test
    data_set_name = "test_4a"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_4b(config, folder_path):
    # run the test
    data_set_name = "test_4b"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res, specific_scenario=True)


def test_4c(config, folder_path):
    # run the test
    data_set_name = "test_4c"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4d(config, folder_path):
    # run the test
    data_set_name = "test_4d"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_5a(config, folder_path):
    # run the test
    data_set_name = "test_5a"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_5b(config, folder_path):
    # run the test
    data_set_name = "test_5b"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_5c(config, folder_path):
    # run the test
    data_set_name = "test_5c"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_5d(config, folder_path):
    # run the test
    data_set_name = "test_5d"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_6a(config, folder_path):
    # run the test
    data_set_name = "test_6a"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_7a(config, folder_path):
    # run the test
    data_set_name = "test_7a"
    config.analysis.objective = "total_carbon_emissions"
    main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_8a(config, folder_path):
    # run the test
    data_set_name = "test_8a"
    main(
        config=config, dataset_path=os.path.join(folder_path, data_set_name)
    )
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    check_get_total_get_full_ts(res)


if __name__ == "__main__":
    from config import config

    config.solver.keep_files = False
    folder_path = os.path.dirname(__file__)
    test_3c(config, folder_path)
