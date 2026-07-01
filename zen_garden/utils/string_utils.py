import logging
import os
import shutil
from pathlib import Path

from zen_garden.utils.scenario_utils import ScenarioUtils


class StringUtils:
    """This class handles strings for logging and filenames to tidy up scripts."""

    def __init__(self):
        """Initializes the class."""
        pass

    @classmethod
    def print_optimization_progress(cls, scenario, steps_horizon, step, system):
        """Prints the current optimization progress.

        :param scenario: string of scenario name
        :param steps_horizon: all steps of horizon
        :param step: current step of horizon
        :param system: system of optimization
        """
        scenario_string = ScenarioUtils.scenario_string(scenario)
        if len(steps_horizon) == 1:
            logging.info(
                f"\n--- Conduct optimization for perfect foresight "
                f"{scenario_string}--- \n"
            )
        else:
            corresponding_year = (
                system.reference_year + step * system.interval_between_years
            )
            logging.info(
                "\n--- Conduct optimization for rolling horizon step for "
                f"{corresponding_year} ({steps_horizon.index(step) + 1} of "
                f"{len(steps_horizon)}) {scenario_string}--- \n"
            )

    @classmethod
    def generate_folder_path(cls, config, scenario, scenario_dict, steps_horizon, step):
        """Generates the folder path for the results.

        :param config: config of optimization
        :param scenario: name of scenario
        :param scenario_dict: current scenario dict
        :param steps_horizon: all steps of horizon
        :param step: current step of horizon
        :return: scenario name in folder
        :return: subfolder in results file
        :return: mapping of parameters
        """
        subfolder = Path("")
        scenario_name = None
        param_map = None
        if config.system.conduct_scenario_analysis:
            # handle scenarios
            scenario_name = f"scenario_{scenario}"
            subfolder = Path(f"scenario_{scenario_dict['base_scenario']}")

            # set the scenarios
            if scenario_dict["sub_folder"] != "":
                # get the param map
                param_map = scenario_dict["param_map"]

                # get the output scenarios
                subfolder = subfolder.joinpath(
                    f"scenario_{scenario_dict['sub_folder']}"
                )
                scenario_name = f"scenario_{scenario_dict['sub_folder']}"

        # handle myopic foresight
        if len(steps_horizon) > 1:
            mf_f_string = f"MF_{step}"
            # handle combination of MF and scenario analysis
            if config.system.conduct_scenario_analysis:
                subfolder = Path(subfolder), Path(mf_f_string)
            else:
                subfolder = Path(mf_f_string)

        return scenario_name, subfolder, param_map

    @classmethod
    def setup_model_folder(cls, analysis, system):
        """Return model name while conducting some tests.

        :param analysis: analysis of optimization
        :param system: system of optimization
        :return: model name
        :return: output folder
        """
        model_name = os.path.basename(analysis.dataset)
        out_folder = cls.setup_output_folder(analysis, system)
        return model_name, out_folder

    @classmethod
    def setup_output_folder(cls, analysis, system):
        """Return model name while conducting some tests.

        :param analysis: analysis of optimization
        :param system: system of optimization
        :return: output folder
        """
        if not os.path.exists(analysis.folder_output):
            try:
                os.mkdir(analysis.folder_output)
            except FileExistsError:
                pass
        out_folder = cls.get_output_folder(analysis)
        if not os.path.exists(out_folder):
            try:
                os.mkdir(out_folder)
            except FileExistsError:
                pass
        else:
            logging.warning(f"The output folder '{out_folder}' already exists")
            if analysis.overwrite_output:
                logging.warning("Existing files will be overwritten!")
                if not system.conduct_scenario_analysis:
                    # TODO fix for scenario analysis, shared folder for all
                    # scenarios, so not robust for parallel process
                    for filename in os.listdir(out_folder):
                        file_path = os.path.join(out_folder, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
        return out_folder

    @staticmethod
    def get_output_folder(analysis):
        """Return name of output folder.

        :param analysis: analysis of optimization
        :return: output folder
        """
        model_name = os.path.basename(analysis.dataset)
        out_folder = os.path.join(analysis.folder_output, model_name)
        return out_folder
