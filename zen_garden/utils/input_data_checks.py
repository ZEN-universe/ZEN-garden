import logging
import os
import warnings

import numpy as np

from zen_garden.default_config import System
from zen_garden.services.dataset_path_resolver import DatasetPathResolver

logger = logging.getLogger(__name__)

ATTRIBUTE_FILENAMES = ("attributes.json", "attributes.yaml", "attributes.yml")


class InputDataChecks:
    """This class checks if the input data (folder/file structure, system.py settings,
    element definitions, etc.) is defined correctly.
    """

    dataset_path_resolver: "DatasetPathResolver | None" = None

    def __init__(self, model_schema):
        """Initialize the class.

        Args:
            model_schema: model schema exposing the canonical configuration used
                to extract the analysis, system and solver dictionaries
        """
        self.model_schema = model_schema

    @property
    def config(self):
        return self.model_schema.config

    @property
    def system(self):
        return self.model_schema.config.system

    @system.setter
    def system(self, value):
        self.model_schema.config.system = value

    @property
    def analysis(self):
        return self.model_schema.config.analysis

    @analysis.setter
    def analysis(self, value):
        self.model_schema.config.analysis = value

    def check_primary_folder_structure(self):
        """Checks if the primary folder structure (set_conversion_technology,
        set_transport_technology, ..., energy_system) is provided correctly.
        """
        for set_name, subsets in self.analysis.subsets.model_dump().items():
            if not os.path.exists(os.path.join(self.analysis.dataset, set_name)):
                raise AssertionError(f"Folder {set_name} does not exist!")
            if isinstance(subsets, dict):
                for subset_name, _subset in subsets.items():
                    if not os.path.exists(
                        os.path.join(self.analysis.dataset, set_name, subset_name)
                    ):
                        raise AssertionError(f"Folder {subset_name} does not exist!")
                else:
                    for subset_name in subsets:
                        if not os.path.exists(
                            os.path.join(self.analysis.dataset, set_name, subset_name)
                        ):
                            raise AssertionError(
                                f"Folder {subset_name} does not exist!"
                            )

        energy_system_files = os.listdir(
            os.path.join(self.analysis.dataset, "energy_system")
        )
        if not any(name in energy_system_files for name in ATTRIBUTE_FILENAMES):
            raise FileNotFoundError(
                "An attributes file is missing in the energy_system directory. "
                f"Expected one of: {', '.join(ATTRIBUTE_FILENAMES)}"
            )

        base_unit_filenames = (
            "base_units.yaml",
            "base_units.yml",
            "base_units.json",
        )
        if not any(name in energy_system_files for name in base_unit_filenames):
            raise FileNotFoundError(
                "A base-units file is missing in the energy_system directory. "
                f"Expected one of: {', '.join(base_unit_filenames)}"
            )

        for file_name in [
            "set_edges.csv",
            "set_nodes.csv",
            "unit_definitions.txt",
        ]:
            if (
                file_name not in energy_system_files
                and file_name.replace(".csv", ".json") not in energy_system_files
            ):
                raise FileNotFoundError(
                    f"File {file_name} is missing in the energy_system directory"
                )

    def check_existing_technology_data(self):
        """This method checks the existing technology input data and only regards
        those technology elements whose folders contain a supported attributes file.
        """
        assert (
            self.config is not None and self.dataset_path_resolver is not None
        ), "Config and dataset path resolver must be set before calling this method"
        # TODO works for two levels of subsets, but not for more
        self.config.system.set_technologies = []
        for set_name, subsets in self.config.analysis.subsets[
            "set_technologies"
        ].items():
            for technology in self.config.system[set_name]:
                if technology not in self.dataset_path_resolver.elements_of_set(
                    set_name
                ):
                    # raise error if technology is not in input data
                    raise FileNotFoundError(
                        f"Technology {technology} selected in config does not "
                        "exist in input data"
                    )
                elif not self._has_attribute_file(set_name, technology):
                    raise FileNotFoundError(
                        "No supported attributes file exists for the "
                        f"technology {technology}"
                    )
            self.config.system.set_technologies.extend(self.config.system[set_name])
            # check subsets of technology_subset
            assert isinstance(
                subsets, list
            ), f"Subsets of {set_name} must be a list, dict not implemented"
            for subset in subsets:
                for technology in self.config.system[subset]:
                    if technology not in self.dataset_path_resolver.elements_of_set(
                        subset
                    ):
                        # raise error if technology is not in input data
                        raise FileNotFoundError(
                            f"Technology {technology} selected in config does "
                            "not exist in input data"
                        )
                    elif not self._has_attribute_file(subset, technology):
                        raise FileNotFoundError(
                            "No supported attributes file exists for the "
                            f"technology {technology}"
                        )
                    self.config.system[set_name].extend(self.config.system[subset])
                    self.config.system.set_technologies.extend(
                        self.config.system[subset]
                    )

    def check_existing_carrier_data(self, carriers: list[str]):
        """Checks the existing carrier data and only regards those carriers for
        which folders exist.
        """
        assert self.dataset_path_resolver is not None
        # check if carriers exist
        for carrier in carriers:
            if carrier not in self.dataset_path_resolver.elements_of_set(
                "set_carriers"
            ):
                # raise error if carrier is not in input data
                raise FileNotFoundError(
                    f"Carrier {carrier} selected in config does not exist ininput data"
                )
            elif not self._has_attribute_file("set_carriers", carrier):
                raise FileNotFoundError(
                    f"No supported attributes file exists for the carrier {carrier}"
                )

    def _has_attribute_file(self, set_name: str, element: str) -> bool:
        """Return whether an element has a supported attributes file."""
        assert self.dataset_path_resolver is not None
        paths = self.dataset_path_resolver.paths_of_element(set_name, element)
        return any(filename in paths for filename in ATTRIBUTE_FILENAMES)

    def check_single_directed_edges(self, set_edges_input):
        """Checks if single-directed edges exist in the dataset (e.g. CH-DE exists,
        DE-CH doesn't) and raises a warning.

        Args:
            set_edges_input: DataFrame containing set of edges defined in
                set_edges.csv
        """
        for edge in set_edges_input.values:
            reversed_edge = edge[2] + "-" + edge[1]
            if (
                reversed_edge
                not in [edge_string[0] for edge_string in set_edges_input.values]
                and edge[1] in self.system.set_nodes
                and edge[2] in self.system.set_nodes
            ):
                warnings.warn(
                    f"The edge {edge[0]} is single-directed, i.e., the edge "
                    f"{reversed_edge} doesn't exist!",
                    stacklevel=2,
                )

    def read_system_file(self, config):
        """Reads the system file and updates the config instance.

        Args:
            config: config object containing the dataset path and current system
                settings.
        """
        system_path = None
        for filename in ["system.yaml", "system.yml", "system.json"]:
            candidate = os.path.join(config.analysis.dataset, filename)
            if os.path.exists(candidate):
                system_path = candidate
                break

        if system_path is None:
            raise FileNotFoundError(
                f"No system definition file found in dataset "
                f"'{config.analysis.dataset}'. "
                f"Expected one of: system.yaml, system.yml, system.json."
            )

        system = System.from_file(system_path)
        new_system = config.system.model_copy(update=system.model_dump())
        config.system = new_system
        self.system = new_system

    def check_carrier_configuration(
        self, input_carrier, output_carrier, reference_carrier, name
    ):
        """Check the chosen input/output/reference carrier combination.

        :param input_carrier: input carrier of conversion technology
        :param output_carrier: output carrier of conversion technology
        :param reference_carrier: reference carrier of technology
        :param name: name of conversion technology
        """
        # assert that conversion technology has at least an input/output carrier
        assert (
            len(input_carrier + output_carrier) > 0
        ), f"Conversion technology {name} has neither an input nor an output carrier!"
        # check if reference carrier in input and output carriers and set
        # technology to correspondent carrier
        assert reference_carrier[0] in (input_carrier + output_carrier), (
            f"reference carrier {reference_carrier} of technology {name} not "
            f"in input and output carriers {input_carrier + output_carrier}"
        )
        set_input_carrier = set(input_carrier)
        set_output_carrier = set(output_carrier)
        # assert that input and output carrier of conversion tech are different
        common_carriers = set_input_carrier & set_output_carrier
        assert not common_carriers, (
            f"The conversion technology {name} has the same input and output "
            f"carrier(s) ({list(common_carriers)})!"
        )

    def check_duplicate_indices(self, df_input, file_name, folder_path):
        """Checks if df_input contains any duplicate indices and either removes
        them if they are of identical value or raises an error otherwise.

        :param df_input: raw input dataframe
        :param folder_path: the path of the folder containing the selected file
        :param file_name: name of selected file
        :return: df_input without duplicate indices
        """
        unique_elements, counts = np.unique(df_input.index, return_counts=True)
        duplicates = unique_elements[counts > 1]

        if len(duplicates) != 0:
            for duplicate in duplicates:
                values = df_input.loc[duplicate]
                # check if all the duplicates are of the same value
                if values.nunique() == 1:
                    logger.warning(
                        f"The input data file {file_name + '.csv'} at "
                        f"{folder_path} contains duplicate indices with "
                        f"identical values: {df_input.loc[duplicates]}."
                    )
                else:
                    raise AssertionError(
                        f"The input data file {file_name + '.csv'} at "
                        f"{folder_path} contains duplicate indices with "
                        f"different values: {df_input.loc[duplicates]}."
                    )
            # remove duplicates
            duplicate_mask = df_input.index.duplicated(keep="first")
            df_input = df_input[~duplicate_mask]

        return df_input
