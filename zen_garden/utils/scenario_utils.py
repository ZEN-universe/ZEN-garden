import json
import logging
import os
import shutil
from collections import defaultdict

import yaml

from zen_garden.services.scenario_dict import ScenarioDict

logger = logging.getLogger(__name__)


class ScenarioUtils:
    """This class handles some stuff for scenarios to tidy up scripts."""

    def __init__(self):
        """Initializes the class."""
        pass

    @staticmethod
    def scenario_string(scenario):
        """Creates additional scenario string.

        :param scenario: scenario name
        :return: scenario string
        """
        if scenario != "":
            scenario_string = f"for scenario {scenario} "
        else:
            scenario_string = ""
        return scenario_string

    @staticmethod
    def clean_scenario_folder(config, out_folder):
        """Cleans scenario dict when overwritten.

        :param config: config of optimization
        :param out_folder: output folder
        """
        # compare to existing sub-scenarios
        if (
            config.system.conduct_scenario_analysis
            and config.system.clean_sub_scenarios
        ):
            # collect all paths that are in the scenario dict
            folder_dict = defaultdict(list)
            for _key, value in config.scenarios.items():
                if value["sub_folder"] != "":
                    folder_dict[f"scenario_{value['base_scenario']}"].append(
                        f"scenario_{value['sub_folder']}"
                    )
                    folder_dict[f"scenario_{value['base_scenario']}"].append(
                        f"dict_all_sequence_time_steps_{value['sub_folder']}.h5"
                    )
            for scenario_name, sub_folders in folder_dict.items():
                scenario_path = os.path.join(out_folder, scenario_name)
                if os.path.exists(scenario_path) and os.path.isdir(scenario_path):
                    existing_sub_folder = os.listdir(scenario_path)
                    for sub_folder in existing_sub_folder:
                        # delete the scenario subfolder
                        sub_folder_path = os.path.join(scenario_path, sub_folder)
                        if (
                            os.path.isdir(sub_folder_path)
                            and sub_folder not in sub_folders
                        ):
                            logger.info(f"Removing sub-scenario {sub_folder}")
                            shutil.rmtree(sub_folder_path, ignore_errors=True)
                        # the time steps dict
                        if (
                            sub_folder.startswith("dict_all_sequence_time_steps")
                            and sub_folder not in sub_folders
                        ):
                            logger.info(f"Removing time steps dict {sub_folder}")
                            os.remove(sub_folder_path)

    @staticmethod
    def get_scenarios(config, job_index):
        """Retrieves and overwrites the scenario dicts.

        :param config: config of optimization
        :param job_index: index of current job, if passed
        :return: scenarios of optimization
        :return: elements in scenario
        """
        if config.system.conduct_scenario_analysis:
            scenarios_path = None
            for filename in ["scenarios.yaml", "scenarios.yml", "scenarios.json"]:
                candidate = os.path.join(config.analysis.dataset, filename)
                if os.path.isfile(candidate):
                    scenarios_path = os.path.abspath(candidate)
                    break

            if scenarios_path is None:
                raise FileNotFoundError(
                    f"No scenarios file found in dataset '{config.analysis.dataset}'. "
                    "Expected one of: scenarios.yaml, scenarios.yml, scenarios.json."
                )

            try:
                with open(scenarios_path, "r", encoding="utf-8") as file:
                    if scenarios_path.endswith((".yaml", ".yml")):
                        scenarios = yaml.safe_load(file)
                    else:
                        scenarios = json.load(file)
            except (json.JSONDecodeError, yaml.YAMLError) as exc:
                raise ValueError(
                    f"Failed to parse scenarios file '{scenarios_path}': {exc}"
                ) from exc

            if scenarios is None:
                raise ValueError(
                    f"Scenarios file '{scenarios_path}' is empty or contains no data."
                )

            config.scenarios.update(scenarios)
            # remove the default scenario if necessary
            if not config.system.run_default_scenario and "" in config.scenarios:
                del config.scenarios[""]

            # expand the scenarios
            config.scenarios = ScenarioDict.expand_lists(config.scenarios)

            # deal with the job array
            if job_index is not None:
                if isinstance(job_index, int):
                    job_index = [job_index]
                else:
                    job_index = list(job_index)
                logger.info(f"Running scenarios with job indices: {job_index}")
                # reduce the scenario and element to a single one
                scenarios = [list(config.scenarios.keys())[jx] for jx in job_index]
                elements = [list(config.scenarios.values())[jx] for jx in job_index]
            else:
                logger.info("Running all scenarios sequentially")
                scenarios = config.scenarios.keys()
                elements = config.scenarios.values()
        # Nothing to do with the scenarios
        else:
            scenarios = [""]
            elements = [{}]
        return scenarios, elements
