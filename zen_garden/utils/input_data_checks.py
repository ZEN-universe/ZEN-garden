import importlib.util
import json
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

from zen_garden.default_config import Subscriptable

if TYPE_CHECKING:
    from zen_garden.model.config import Config
    from zen_garden.model.context import Context


class InputDataChecks:
    """This class checks if the input data (folder/file structure, system.py settings,
    element definitions, etc.) is defined correctly.
    """

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
    config: "Config | None" = None
    context: "Context | None" = None
    optimization_setup: object | None = None

    def __init__(self, config):
        """Initialize the class.

        Args:
            config: config object used to extract the analysis, system and solver
                dictionaries
            optimization_setup: OptimizationSetup instance
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
            self.config is not None and self.context is not None
        ), "Config and context must be set before calling this method"
        # TODO works for two levels of subsets, but not for more
        self.config.system.set_technologies = []
        for set_name, subsets in self.config.analysis.subsets[
            "set_technologies"
        ].items():
            for technology in self.config.system[set_name]:
                if technology not in self.context.paths[set_name].keys():
                    # raise error if technology is not in input data
                    raise FileNotFoundError(
                        f"Technology {technology} selected in config does not "
                        "exist in input data"
                    )
                elif "attributes.json" not in self.context.paths[set_name][technology]:
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
                    if technology not in self.context.paths[subset].keys():
                        # raise error if technology is not in input data
                        raise FileNotFoundError(
                            f"Technology {technology} selected in config does "
                            "not exist in input data"
                        )
                    elif (
                        "attributes.json" not in self.context.paths[subset][technology]
                    ):
                        raise FileNotFoundError(
                            "The file attributes.json does not exist for the "
                            "technology {technology}"
                        )
                    self.config.system[set_name].extend(self.config.system[subset])
                    self.config.system.set_technologies.extend(
                        self.config.system[subset]
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
        for char in self.PROHIBITED_DATASET_CHARACTERS:
            if char in dataset:
                raise ValueError(
                    f"Character {char} is not allowed in the dataset name "
                    f"{dataset}\nProhibited characters: "
                    f"{self.PROHIBITED_DATASET_CHARACTERS}"
                )
        # check if chosen dataset contains a system.py file
        if not os.path.exists(
            os.path.join(self.analysis.dataset, "system.py")
        ) and not os.path.exists(os.path.join(self.analysis.dataset, "system.json")):
            raise FileNotFoundError(
                f"Neither system.json nor system.py not found in dataset: "
                f"{self.analysis.dataset}"
            )

    def read_system_file(self, config):
        """Reads the system file and returns the system dictionary.

        :param config: config object
        """
        # check if system.json file exists
        if os.path.exists(os.path.join(config.analysis.dataset, "system.json")):
            with open(
                os.path.join(config.analysis.dataset, "system.json"), "r"
            ) as file:
                system = json.load(file)
        # otherwise read system.py file
        else:
            system_path = os.path.join(config.analysis.dataset, "system.py")
            spec = importlib.util.spec_from_file_location("module", system_path)
            assert spec is not None, f"Could not load system.py from {system_path}"
            module = importlib.util.module_from_spec(spec)
            assert (
                spec.loader is not None
            ), f"Could not load system.py from {system_path}. spec.loader is None."
            spec.loader.exec_module(module)
            system = module.system
        new_system = config.system.model_copy(update=system)
        config.system = new_system
        self.system = new_system
        self.check_no_extra_config_fields(config)

    def check_no_extra_config_fields(self, config, config_name="config"):
        """Checks if the config object has no extra fields that are not defined
        in the default_config.
        """
        assert len(config.model_extra) == 0, (
            f"The config object '{config_name}' has extra fields that are not "
            f"defined in the default_config: {config.model_extra}."
        )
        for name in config.__class__.model_fields:
            subconfig = getattr(config, name)
            # Detect if the subconfig is a subclass of Subscriptable
            if isinstance(subconfig.__class__, type) and issubclass(
                subconfig.__class__, Subscriptable
            ):
                self.check_no_extra_config_fields(
                    subconfig, config_name=config_name + "/" + name
                )
