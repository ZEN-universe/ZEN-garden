"""
File that contains functions to compare the results of two or more models.
"""
from zen_garden.postprocess.results import Results
from zen_garden.postprocess.results.solution_loader import ComponentType
from typing import Optional, Any
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np


def compare_model_values(
    results: list[Results],
    component_type: ComponentType | str,
    compare_total: bool = True,
    scenarios: list[str] = [],
) -> dict[str, pd.DataFrame]:
    """Compares the input data of two or more results

    :param results: list of results
    :param component_type: component type to compare
    :param compare_total: if True, compare total value, not full time series
    :param scenarios: None, str or tuple of scenarios
    :return: a dictionary with diverging results
    """

    scenarios = check_and_fill_scenario_list(results, scenarios)

    if isinstance(component_type, str):
        component_type = ComponentType(component_type)

    logging.info(
        f"Comparing the model parameters of {results[0].solution_loader.name, results[1].solution_loader.name} and scenarios {scenarios[0], scenarios[1]}"
    )

    diff_components = get_component_diff(results, component_type)

    diff_dict = {}
    # initialize progress bar
    pbar = tqdm(total=len(diff_components))
    for component_name in diff_components:
        # update progress bar
        pbar.update(1)
        pbar.set_description(f"Compare parameter {component_name}")

        comparison_df = compare_component_values(
            results, component_name, compare_total, scenarios
        )

        if not comparison_df.empty:
            logging.info(f"Parameter {component_name} has different values")
            diff_dict[component_name] = comparison_df
    pbar.close()
    return diff_dict


def compare_configs(
    results: list[Results],
    scenario_name: str,
) -> dict[str, Any]:
    """
    Compares the configs of two results, namely the Analysis-Config and the System-config.

    :param results: List of results
    :param scenario_name: List of scenarios to filter by
    :return: dictionary with diverging configs
    """
    ans: dict[str, Any] = {}

    scenario_names = [scenario_name, scenario_name]

    for i in range(2):
        if scenario_names[i] not in results[i].solution_loader.scenarios:
            random_scenario = next(iter(results[i].solution_loader.scenarios.keys()))
            logging.info(
                f"{scenario_name} not in {results[i].solution_loader.name}, choosing {random_scenario}."
            )
            scenario_names[i] = random_scenario

    results_1, results_2 = results
    scenario_name_1, scenario_name_2 = scenario_names

    scenario_1 = results_1.solution_loader.scenarios[scenario_name_1]
    scenario_2 = results_2.solution_loader.scenarios[scenario_name_2]

    names = [results_1.solution_loader.name, results_2.solution_loader.name]

    analysis_diff = compare_dicts(
        scenario_1.analysis.model_dump(), scenario_2.analysis.model_dump(), names
    )

    if analysis_diff is not None:
        ans["analysis"] = analysis_diff

    system_diff = compare_dicts(
        scenario_1.system.model_dump(), scenario_2.system.model_dump(), names
    )

    if system_diff is not None:
        ans["system"] = system_diff

    solver_diff = compare_dicts(
        scenario_1.solver.model_dump(), scenario_2.solver.model_dump(), names
    )

    if solver_diff is not None:
        ans["solver"] = solver_diff

    return ans

def get_component_diff(
    results: list[Results], component_type: ComponentType
) -> list[str]:
    """returns a list with the differences in component names

    :param results: list with results
    :param component_type: component type to compare
    :return: list with the common params
    """
    assert len(results) == 2, "Please give exactly two components"

    results_0, results_1 = results
    component_names_0 = set(
        [
            name
            for name, component in results_0.solution_loader.components.items()
            if component.component_type is component_type
        ]
    )

    component_names_1 = set(
        [
            name
            for name, component in results_1.solution_loader.components.items()
            if component.component_type is component_type
        ]
    )
    only_in_0 = component_names_0.difference(component_names_0)
    only_in_1 = component_names_1.difference(component_names_1)

    common_component = component_names_0.intersection(component_names_1)

    if only_in_0:
        logging.info(
            f"Components {only_in_1} are missing from {results_1.solution_loader.name}"
        )
    elif only_in_1:
        logging.info(
            f"Components {only_in_1} are missing from {results_0.solution_loader.name}"
        )
    return [i for i in common_component]


def compare_dicts(
    dict1: dict[Any, Any],
    dict2: dict[Any, Any],
    result_names: list[str] = ["res_1", "res_2"],
) -> Optional[dict[Any, Any]]:
    """
    Compares two dictionaries and returns only the fields with different values. 

    :param dict1: first config dict
    :param dict2: second config dict
    :param result_names: names of results
    :return: diff dict
    """
    diff_dict = {}
    for key in dict1.keys() | dict2.keys():
        if isinstance(dict1.get(key), dict) and isinstance(dict2.get(key), dict):
            nested_diff = compare_dicts(
                dict1.get(key, {}), dict2.get(key, {}), result_names
            )
            if nested_diff:
                diff_dict[key] = nested_diff
        elif dict1.get(key) != dict2.get(key):
            if isinstance(dict1.get(key), list) and isinstance(dict2.get(key), list):
                if sorted(dict1.get(key)) != sorted(dict2.get(key)):  # type: ignore
                    diff_dict[key] = {
                        result_names[0]: sorted(dict1.get(key)),  # type: ignore
                        result_names[1]: sorted(dict2.get(key)),  # type: ignore
                    }
            else:
                diff_dict[key] = {
                    result_names[0]: dict1.get(key),
                    result_names[1]: dict2.get(key),
                }
    return diff_dict if diff_dict else None


def check_and_fill_scenario_list(
    results: list[Results], scenarios: list[str]
) -> list[str]:
    """Checks if both results have the provided scenarios and otherwise returns a list containing twice one common scenario.

    :param results: List of results.
    :param scenarios: List of names of scenario.
    :return: List of scenario names
    """
    assert len(results) == 2, "Please provide exactly two results"
    assert len(scenarios) <= 2, "Please provide a maximum of two scenarios"

    try:
        common_scenario = get_common_scenario(*results)
    except AssertionError:
        logging.info(
            "No common scenario found. Selecting random scenario for each result."
        )
        scenarios = [
            next(iter(results[0].solution_loader.scenarios.keys())),
            next(iter(results[1].solution_loader.scenarios.keys())),
        ]

    if len(scenarios) == 0:
        scenarios.append(common_scenario)

    if len(scenarios) == 1:
        if scenarios[0] in results[1].solution_loader.scenarios:
            scenarios.append(scenarios[0])
        else:
            scenarios.append(next(iter(results[1].solution_loader.scenarios.keys())))

    for i in range(2):
        assert (
            scenarios[i] in results[i].solution_loader.scenarios
        ), f"{scenarios[i]} not in {results[i].solution_loader.scenarios.keys()}"

    return scenarios


def get_common_scenario(results_1: Results, results_2: Results) -> str:
    """
    Returns the name of a common scenario that are in both provided results.

    :param results_1: Results 1
    :param results_2: Results 2
    :return: Name of common scenario
    """
    common_scenarios = set(results_1.solution_loader.scenarios.keys()).intersection(
        results_2.solution_loader.scenarios.keys()
    )
    assert len(common_scenarios) > 0, "No common scenarios between provided scenarios."

    return next(iter(common_scenarios))


def compare_component_values(
    results: list[Results],
    component_name: str,
    compare_total: bool,
    scenarios: list[str] = [],
    rtol: float = 1e-3,
) -> pd.DataFrame:
    """Compares component values of two results

    :param results: list with results
    :param component_name: component name
    :param compare_total: if True, compare total value, not full time series
    :param scenarios: None, str or tuple of scenarios
    :param rtol: relative tolerance of equal values
    :return: dictionary with diverging component values
    """
    scenarios = check_and_fill_scenario_list(results, scenarios)

    result_names = [result.solution_loader.name for result in results]

    results_0, results_1 = results
    scenario_0, scenario_1 = scenarios

    if compare_total:
        val_0 = results_0.get_total(component_name, scenario_name=scenario_0)
        val_1 = results_1.get_total(component_name, scenario_name=scenario_1)
    else:
        val_0 = results_0.get_full_ts(component_name, scenario_name=scenario_0)
        val_1 = results_1.get_full_ts(component_name, scenario_name=scenario_1)
    return _get_comparison_df(val_0, val_1, result_names, component_name, rtol)

def _get_comparison_df(val_0, val_1, result_names, component_name, rtol):
    """
    :param val_0:
    :param val_1:
    :param result_names:
    :param component_name:
    :param rtol:
    :return:
    """
    mismatched_index = False
    mismatched_shape = False
    if isinstance(val_0, pd.DataFrame):
        val_0 = val_0.sort_index(axis=0).sort_index(axis=1)
    else:
        val_0 = val_0.sort_index()
    if isinstance(val_1, pd.DataFrame):
        val_1 = val_1.sort_index(axis=0).sort_index(axis=1)
    else:
        val_1 = val_1.sort_index()
    if val_1.shape == val_0.shape:
        if len(val_0.shape) == 2:
            if not val_0.index.equals(val_1.index) or not val_0.columns.equals(
                val_1.columns
            ):
                mismatched_index = True
        elif not val_0.index.equals(val_1.index):
            mismatched_index = True
    else:
        logging.info(
            f"Component {component_name} changed shape from {val_0.shape} ({result_names[0]}) to {val_1.shape} ({result_names[1]})"
        )
        mismatched_shape = True
        if not val_0.index.equals(val_1.index):
            mismatched_index = True

    if mismatched_index:
        logging.info(
            f"Component {component_name} does not have matching index or columns"
        )
        missing_index = val_0.index.difference(val_1.index) if len(val_0.index) > len(val_1.index) else val_1.index.difference(val_0.index)
        common_index = val_0.index.intersection(val_1.index)
        val_0_aligned = val_0.loc[common_index]
        val_1_aligned = val_1.loc[common_index]
        comparison_df_aligned = _get_different_vals(val_0_aligned, val_1_aligned, result_names, rtol)
        wrong_index = comparison_df_aligned.index.union(missing_index)
        comparison_df = pd.concat([val_0, val_1], keys=result_names, axis=1)
        comparison_df = comparison_df.sort_index(axis=1, level=1)
        comparison_df = comparison_df.loc[wrong_index]
        return comparison_df
    # if not mismatched_shape:
    if not mismatched_shape:
        return _get_different_vals(val_0, val_1, result_names, rtol)
    else:
        # check if the dataframe has only constant values along axis 1
        if isinstance(val_0, pd.DataFrame) and (val_0.nunique(axis=1) == 1).all():
            val_0 = val_0.iloc[:, 0]
        elif isinstance(val_1, pd.DataFrame) and (val_1.nunique(axis=1) == 1).all():
            val_1 = val_1.iloc[:, 0]
        else:
            logging.info(f"Component {component_name} has different values")
            comparison_df = pd.concat([val_0, val_1], keys=result_names, axis=1)
            comparison_df = comparison_df.sort_index(axis=1, level=1)
            return comparison_df
        return _get_different_vals(val_0, val_1, result_names, rtol)

def _get_different_vals(
    val_0: "pd.DataFrame | pd.Series[Any]",
    val_1: "pd.DataFrame | pd.Series[Any]",
    result_names: list[str],
    rtol: float,
) -> pd.DataFrame:
    """
    Get the different values of two dataframes or series
    
    :param val_0: first dataframe or series
    :param val_1: second dataframe or series
    :param result_names: names of the results
    :param rtol: relative tolerance of equal values
    :return: comparison_df
    """
    is_close = np.isclose(val_0, val_1, rtol=rtol, equal_nan=True)
    if isinstance(val_0, pd.DataFrame):
        diff_val_0 = val_0[(~is_close).any(axis=1)]
        diff_val_1 = val_1[(~is_close).any(axis=1)]
    else:
        diff_val_0 = val_0[(~is_close)]
        diff_val_1 = val_1[(~is_close)]
    comparison_df = pd.concat([diff_val_0, diff_val_1], keys=result_names, axis=1)
    comparison_df = comparison_df.sort_index(axis=1, level=1)
    return comparison_df
