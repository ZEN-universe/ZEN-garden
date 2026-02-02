import json
import os
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest

from zen_garden import run, Results, download_example_dataset
from zen_garden.wrapper.operation_scenarios import operation_scenarios


# fixtures
##########

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
                scenario = results.solution_loader.scenarios[s]
                test_values = test_variables[test_model][s]
                for c in test_values:
                    if c in scenario.components:
                        values = results.get_df(c,scenario_name=s)
                        for test_value in test_values[c]:
                            if isinstance(test_value["index"],list):
                                if len(test_value["index"]) == 1:
                                    test_index = test_value["index"][0]
                                else:
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
    year=None,
    discount_to_first_step=True,
    get_doc=False,
):
    """
    Tests the functionality of the Results methods get_total() and get_full_ts()

    :param get_doc:
    :param discount_to_first_step: Apply annuity to first year of interval or entire interval
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
    if get_doc:
        results.get_doc(test_variables[0])


# All the tests
###############
def test_1a(folder_path):
    # add duals for this test

    # test also whether config and dataset can take just file name in cwd
    cwd = os.getcwd()
    os.chdir(folder_path)
   
    # run the test
    data_set_name = "test_1a"
    run(
        config=os.path.join("config_duals.json"),
        dataset=os.path.join(data_set_name),
    )

    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)
    os.chdir(cwd)


def test_1b(folder_path):
    # run the test
    data_set_name = "test_1b"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1c(folder_path):
    # run the test
    data_set_name = "test_1c"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1d(folder_path):
    # run the test
    data_set_name = "test_1d"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1e(folder_path):
    # run the test
    data_set_name = "test_1e"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1f(folder_path):
    # run the test
    data_set_name = "test_1f"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1g(folder_path):
    # run the test
    data_set_name = "test_1g"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)

def test_1h(folder_path):
    # run the test
    data_set_name = "test_1h"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1i(folder_path):
    # run the test
    data_set_name = "test_1i"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)

def test_1j(folder_path):
    # run the test
    data_set_name = "test_1j"
    data_set_name_op = data_set_name + "_none__operation"
    run(
        config=os.path.join(folder_path,"config_duals.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs"),
    )
    operation_scenarios(
        config=os.path.join(folder_path,"config_duals.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs"),
        delete_data="True"
    )

    # read the results and check again
    res_cap = Results(os.path.join(folder_path, "outputs", data_set_name))
    res_op = Results(os.path.join(folder_path, "outputs", data_set_name_op))
    compare_variables_results(data_set_name + "_capacity", res_cap, folder_path)
    compare_variables_results(data_set_name + "_operation", res_op, folder_path)

def test_2a(folder_path):
    # run the test
    data_set_name = "test_2a"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2b(folder_path):
    # run the test
    data_set_name = "test_2b"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2c(folder_path):
    # run the test
    data_set_name = "test_2c"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3a(folder_path):
    # run the test
    data_set_name = "test_3a"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3b(folder_path):
    # run the test
    data_set_name = "test_3b"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_3c(folder_path):
    # run the test
    data_set_name = "test_3c"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res, year=2022)


def test_3d(folder_path):
    # run the test
    data_set_name = "test_3d"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # compare the variables of the optimization setup ## disabled for myopic foresight tests!
    # compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res, discount_to_first_step=False)


def test_3e(folder_path):
    # run the test
    data_set_name = "test_3e"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # compare the variables of the optimization setup ## disabled for myopic foresight tests!
    # compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3f(folder_path):
    # run the test
    data_set_name = "test_3f"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # compare the variables of the optimization setup ## disabled for myopic foresight tests!
    # compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3g(folder_path):
    # run the test
    data_set_name = "test_3g"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_3h(folder_path):
    # run the test
    data_set_name = "test_3h"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_3i(folder_path):
    # run the test
    data_set_name = "test_3i"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)

def test_4a(folder_path):
    # run the test
    data_set_name = "test_4a"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_4b(folder_path):
    # run the test
    data_set_name = "test_4b"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res, specific_scenario=True)


def test_4c(folder_path):
    # run the test
    data_set_name = "test_4c"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4d(folder_path):
    # run the test
    data_set_name = "test_4d"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_5a(folder_path):
    # run the test
    data_set_name = "test_5a"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_5b(folder_path):
    # run the test
    data_set_name = "test_5b"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_5c(folder_path):
    # run the test
    data_set_name = "test_5c"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_5d(folder_path):
    # run the test
    data_set_name = "test_5d"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_6a(folder_path):
    # run the test
    data_set_name = "test_6a"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_7a(folder_path):
    # run the test
    data_set_name = "test_7a"
    run(
        config=os.path.join(folder_path,"config_objective.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )

    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)

def test_7b(folder_path):
    # run the test
    data_set_name = "test_7b"
    data_set_name_op = data_set_name + "_none__operation"
    run(
        config=os.path.join(folder_path,"config_duals.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs"),
    )
    operation_scenarios(
        config=os.path.join(folder_path,"config_duals.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs"),
        delete_data=True
    )
    # read the results and check again
    res_cap = Results(os.path.join(folder_path, "outputs", data_set_name))
    res_op = Results(os.path.join(folder_path, "outputs", data_set_name_op))
    compare_variables_results(data_set_name + "_capacity", res_cap, folder_path)
    compare_variables_results(data_set_name + "_operation", res_op, folder_path)

def test_8a(folder_path):
    # run the test
    data_set_name = "test_8a"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )
    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    check_get_total_get_full_ts(res)


def test_9a(folder_path):
    # run the test
    data_set_name = "test_9a"
    with pytest.raises(AssertionError, match='The attribute units defined in the energy_system are not consistent!'):
        run(
            config=os.path.join(folder_path,"config.json"),
            dataset=os.path.join(folder_path,data_set_name),
            folder_output=os.path.join(folder_path,"outputs")
        )

def test_10a(folder_path):
    # run the test
    data_set_name = "test_10a"
    run(
        config=os.path.join(folder_path,"config.json"),
        dataset=os.path.join(folder_path,data_set_name),
        folder_output=os.path.join(folder_path,"outputs")
    )
    # read the results and check again
    res = Results(os.path.join(folder_path, "outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


if __name__ == "__main__":
    folder_path = os.path.dirname(__file__)
    test_7b(folder_path)
