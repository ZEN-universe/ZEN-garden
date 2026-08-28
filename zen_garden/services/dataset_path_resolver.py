import os
from typing import TYPE_CHECKING, Any

from zen_garden.default_config import Subsets

if TYPE_CHECKING:
    from zen_garden.topology.model_schema import ModelSchema


FOLDER_KEY: str = "folder"


class DatasetPathResolver:
    """This class resolves the paths of the dataset based on the configuration."""

    def __init__(self, model_schema: "ModelSchema"):
        self.model_schema = model_schema
        self._paths = self._resolve_dataset_paths()

    @property
    def _config(self):
        return self.model_schema.config

    def _resolve_dataset_paths(self) -> dict:
        """This method creates a dictionary with the paths of the data split
        by carriers, networks, technologies.

        :return: dictionary with paths
        """
        # define path to access dataset related to the current analysis
        paths = {}
        path_data = self._config.analysis.dataset
        assert os.path.exists(
            self._config.analysis.dataset
        ), f"Folder for input data {self._config.analysis.dataset} does not exist!"

        # create a dictionary with the keys based on the folders in path_data
        for folder_name in next(os.walk(path_data))[1]:
            paths[folder_name] = {FOLDER_KEY: os.path.join(path_data, folder_name)}

        # add element paths and their file paths
        self._process_subsets(paths, dict(self._config.analysis.subsets.items()))

        return paths

    def _process_subsets(self, paths: dict, subsets_dict: dict[str, Any]) -> None:
        """Recursively process the subsets dictionary to add folder paths.

        :param paths: Dictionary of paths to update
        :param subsets_dict: Dictionary of subsets to process"""
        for set_name, subsets in subsets_dict.items():
            path = paths[set_name][FOLDER_KEY]

            if isinstance(subsets, dict):
                self._add_folder_paths(paths, set_name, path, list(subsets.keys()))
                self._process_subsets(paths, subsets)
                continue

            self._add_folder_paths(paths, set_name, path, subsets)
            for element in subsets:
                if self._config.system[element]:
                    self._add_folder_paths(paths, element, paths[element][FOLDER_KEY])

    def _add_folder_paths(
        self,
        paths: dict[str, dict[str, dict[str, str]]],
        set_name: str,
        path,
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
                paths[set_name][element] = {FOLDER_KEY: os.path.join(path, element)}
                sub_path = os.path.join(path, element)
                for file in next(os.walk(sub_path))[2]:
                    paths[set_name][element][file] = os.path.join(sub_path, file)
                # add element paths to parent sets
                parent_sets = self._find_parent_set(
                    self._config.analysis.subsets, set_name
                )
                for parent_set in parent_sets:
                    paths[parent_set][element] = paths[set_name][element]
            else:
                paths[element] = {FOLDER_KEY: os.path.join(path, element)}

    def _find_parent_set(
        self, dictionary: Subsets | dict[str, list[str]], subset, path=None
    ):
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
                result = self._find_parent_set(value, subset, current_path)
                if result:
                    return result
        return []

    def all_sets(self):
        """This method returns the sets of the dataset.

        :return: list of sets
        """
        return self._paths.keys()

    def elements_of_set(self, set: str):
        """This method returns the elements of a set.

        :return: list of elements
        """
        return self._paths[set].keys()

    def folder_of_set(self, set: str) -> str:
        """This method returns the folder of a set.

        :return: folder path
        """
        return self._paths[set][FOLDER_KEY]

    def paths_of_element(self, set: str, element: str) -> list[str]:
        """This method returns the path of an element.

        :return: path of element
        """
        return self._paths[set][element]

    def folder_of_element(self, set: str, element: str) -> str:
        """This method returns the folder of an element.

        :return: folder of element
        """
        return self._paths[set][element][FOLDER_KEY]
