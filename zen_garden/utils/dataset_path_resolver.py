import os
from typing import TYPE_CHECKING, Any

from zen_garden.default_config import Subsets

if TYPE_CHECKING:
    from zen_garden.model.config import Config


def resolve_dataset_paths(config: "Config") -> dict:
    """This method creates a dictionary with the paths of the data split
    by carriers, networks, technologies.

    :return: dictionary with paths
    """
    # define path to access dataset related to the current analysis
    paths = {}
    path_data = config.analysis.dataset
    assert os.path.exists(
        config.analysis.dataset
    ), f"Folder for input data {config.analysis.dataset} does not exist!"

    # create a dictionary with the keys based on the folders in path_data
    for folder_name in next(os.walk(path_data))[1]:
        paths[folder_name] = {"folder": os.path.join(path_data, folder_name)}

    # add element paths and their file paths
    _process_subsets(paths, dict(config.analysis.subsets.items()), config)

    return paths


def _process_subsets(
    paths: dict, subsets_dict: dict[str, Any], config: "Config"
) -> None:
    """Recursively process the subsets dictionary to add folder paths.

    :param paths: Dictionary of paths to update
    :param subsets_dict: Dictionary of subsets to process"""
    for set_name, subsets in subsets_dict.items():
        path = paths[set_name]["folder"]

        if isinstance(subsets, dict):
            _add_folder_paths(paths, set_name, path, config, list(subsets.keys()))
            _process_subsets(paths, subsets, config)
            continue

        _add_folder_paths(paths, set_name, path, config, subsets)
        for element in subsets:
            if config.system[element]:
                _add_folder_paths(paths, element, paths[element]["folder"], config)


def _add_folder_paths(
    paths: dict[str, dict[str, dict[str, str]]],
    set_name: str,
    path,
    config: "Config",
    subsets=None,
):
    """Add file paths of element to paths dictionary.

    :param paths: dictionary of paths
    :param set_name: name of set
    :param path: path to folder
    :param subsets: list of subsets
    """
    if subsets is None:
        subsets = []
    for element in next(os.walk(path))[1]:
        if element not in subsets:
            paths[set_name][element] = {"folder": os.path.join(path, element)}
            sub_path = os.path.join(path, element)
            for file in next(os.walk(sub_path))[2]:
                paths[set_name][element][file] = os.path.join(sub_path, file)
            # add element paths to parent sets
            parent_sets = _find_parent_set(config.analysis.subsets, set_name)
            for parent_set in parent_sets:
                paths[parent_set][element] = paths[set_name][element]
        else:
            paths[element] = {"folder": os.path.join(path, element)}


def _find_parent_set(dictionary: Subsets | dict[str, list[str]], subset, path=None):
    """This method finds the parent sets of a subset.

    :param dictionary: dictionary of subsets
    :param subset: subset to find parent sets of
    :param path: path to subset
    :return: list of parent sets
    """
    path = path if path is not None else []
    for key, value in dictionary.items():
        current_path = path + [key]
        if subset in value:
            return current_path
        elif isinstance(value, dict):
            result = _find_parent_set(value, subset, current_path)
            if result:
                return result
    return []
