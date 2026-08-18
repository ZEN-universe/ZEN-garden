import logging
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

from zen_garden.default_config import System
from zen_garden.services.dataset_path_resolver import DatasetPathResolver

if TYPE_CHECKING:
    from zen_garden.model.config import Config

logger = logging.getLogger(__name__)

PROHIBITED_DATASET_CHARACTERS = [
    " ",
    ".",
    ":",
    ",",
    ";",
    "!",
    "?",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "&",
    "|",
    "*",
    "^",
    "%",
    "$",
    "#",
    "@",
    "`",
    "~",
    "\\",
    "/",
]


class InputDataChecks:
    """This class checks if the input data (folder/file structure, system.py settings,
    element definitions, etc.) is defined correctly.
    """

    config: "Config | None" = None
    dataset_path_resolver: "DatasetPathResolver | None" = None

    def __init__(self, config):
        """Initialize the class.

        Args:
            config: config object used to extract the analysis, system and solver
                dictionaries
        """
        self.system = config.system
        self.analysis = config.analysis

    def check_technology_selections(self):
        """Checks selection of different technologies in system.py file."""
        # Checks if at least one technology is selected in the system.py file
        assert (
            len(
                self.system.set_conversion_technologies
                + self.system.set_transport_technologies
                + self.system.set_storage_technologies
            )
            > 0
        ), "No technology selected in system"
        # Checks if identical technologies are selected multiple times in system.py
        # file and removes possible duplicates
        for tech_list in [
            "set_conversion_technologies",
            "set_transport_technologies",
            "set_storage_technologies",
        ]:
            techs_selected = getattr(self.system, tech_list)
            unique_elements = list(np.unique(techs_selected))
            self.system = self.system.model_copy(update={tech_list: unique_elements})

    def check_year_definitions(self):
        """Check if year-related parameters are defined correctly."""
        # assert that number of optimized years is a positive integer
        assert (
            isinstance(self.system.optimized_years, int)
            and self.system.optimized_years > 0
        ), (
            "Number of optimized years must be a positive integer, however it "
            f"is {self.system.optimized_years}"
        )
        # assert that interval between years is a positive integer
        assert (
            isinstance(self.system.interval_between_years, int)
            and self.system.interval_between_years > 0
        ), (
            "Interval between years must be a positive integer, however it is "
            f"{self.system.interval_between_years}"
        )
        assert (
            isinstance(self.system.reference_year, int)
            and self.system.reference_year >= self.analysis.earliest_year_of_data
        ), (
            "Reference year must be an integer and larger than the defined "
            f"earliest_year_of_data: {self.analysis.earliest_year_of_data}"
        )
        # check if the number of years in the rolling horizon isn't larger than
        # the number of optimized years
        if (
            self.system.years_in_rolling_horizon > self.system.optimized_years
            and self.system.use_rolling_horizon
        ):
            warnings.warn(
                "The chosen number of years in the rolling horizon step is "
                "larger than the total number of years optimized!",
                stacklevel=2,
            )

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

        for file_name in [
            "attributes.json",
            "base_units.csv",
            "set_edges.csv",
            "set_nodes.csv",
            "unit_definitions.txt",
        ]:
            if file_name not in os.listdir(
                os.path.join(self.analysis.dataset, "energy_system")
            ) and file_name.replace(".csv", ".json") not in os.listdir(
                os.path.join(self.analysis.dataset, "energy_system")
            ):
                raise FileNotFoundError(
                    f"File {file_name} is missing in the energy_system directory"
                )

    def check_existing_technology_data(self):
        """This method checks the existing technology input data and only regards
        those technology elements for which folders containing the attributes.json
        file exist.
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
                elif (
                    "attributes.json"
                    not in self.dataset_path_resolver.paths_of_element(
                        set_name, technology
                    )
                ):
                    raise FileNotFoundError(
                        "The file attributes.json does not exist for the "
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
                    elif (
                        "attributes.json"
                        not in self.dataset_path_resolver.paths_of_element(
                            subset, technology
                        )
                    ):
                        raise FileNotFoundError(
                            "The file attributes.json does not exist for the "
                            "technology {technology}"
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
            elif "attributes.json" not in self.dataset_path_resolver.paths_of_element(
                "set_carriers", carrier
            ):
                raise FileNotFoundError(
                    f"The file attributes.json does not exist for the carrier {carrier}"
                )

    def check_dataset(self):
        """Ensures that the dataset chosen in the config does exist and contains a
        system.py file.
        """
        dataset = os.path.basename(self.analysis.dataset)
        dirname = os.path.dirname(self.analysis.dataset)
        assert os.path.exists(
            dirname
        ), f"Requested folder {dirname} is not a valid path"
        assert os.path.exists(self.analysis.dataset), (
            f"The chosen dataset {dataset} does not exist at "
            f"{self.analysis.dataset} as it is specified in the config"
        )
        # check if any character in the dataset name is prohibited
        for char in PROHIBITED_DATASET_CHARACTERS:
            if char in dataset:
                raise ValueError(
                    f"Character {char} is not allowed in the dataset name "
                    f"{dataset}\nProhibited characters: "
                    f"{PROHIBITED_DATASET_CHARACTERS}"
                )
        system_files = [
            "system.yaml",
            "system.yml",
            "system.json",
        ]
        if not any(
            os.path.exists(os.path.join(self.analysis.dataset, filename))
            for filename in system_files
        ):
            raise FileNotFoundError(
                f"No system definition file found in dataset "
                f"'{self.analysis.dataset}'. "
                "Expected one of: system.yaml, system.yml, system.json."
            )

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
