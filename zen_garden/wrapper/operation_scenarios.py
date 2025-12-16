from zen_garden import run, Results
from zen_garden.wrapper import utils
from zen_garden.cli.zen_garden_cli import build_parser
from pathlib import Path
import shutil
import os


def operation_scenarios(
    config="./config.json",
    dataset = None,
    folder_output=None,
    job_index=None,
    scenarios_op=None,
    delete_data=False):
    """
    Currently only works if the original problem only has one scenario.

    """

    # clean inputs and set proper default values
    [dataset_path, dataset_name] = os.path.split(Path(dataset))
    
    if folder_output is None:
        folder_output = './outputs'    
    if (job_index is not None) and not isinstance(job_index, list):
        raise TypeError("Job index must be a list of integers")
    
    # extract scenario from the simulation results
    r = Results(Path(folder_output) / dataset_name)
    scenario_list = list(r.solution_loader.scenarios.keys())
    if job_index is not None:
        scenario_list = [scenario_list[i] for i in job_index]

    # run operational scenarios for each scenario in the capacity problem
    for scenario in scenario_list:
        
        # set path of new dataset
        dataset_op = (
            Path(dataset_path) / (dataset_name + "_" + scenario + "__operation")
        )

        # copy original dataset to the new path
        # replace scenarios file with the new operational scenarios
        utils.copy_dataset(
            dataset,
            dataset_op,
            scenarios=scenarios_op
        )

        # extract results on added capacity
        rounding_value = 10**-5
        utils.capacity_addition_2_existing_capacity(
            Path(folder_output) / dataset_name, 
            dataset,
            dataset_op,
            scenario,
            rounding_value
        )

        # turn off capacity_investment
        has_scenarios = (scenarios_op is not None)
        utils.modify_json(
            Path(dataset_op) / "system.json", 
            {
                "allow_investment": False,
                "conduct_scenario_analysis": has_scenarios
            }
        )

        # run operations only simulations
        print("Running Operational Scenarios -----------------------------")
        run(dataset=dataset_op, config=config, folder_output=folder_output)

        # delete created directory
        if delete_data:
            shutil.rmtree(dataset_op)

if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("../../../03_ZEN_data/Reg4Fuels/")
    operation_scenarios(dataset="Reg4Fuels_V9", folder_output="./outputs/Reg4Fuels_V9",
                        job_index=[0])
