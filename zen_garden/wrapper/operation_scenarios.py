from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import shutil
import logging

from zen_garden import run, Results
from zen_garden.wrapper import utils


logger = logging.getLogger(__name__)


def validate_inputs(
    dataset: Path | str,
    folder_output: Path | str | None,
    job_index: Iterable[int] | None
) -> tuple[Path, Path, List[int] | None]:
    """Validate and normalize user-provided inputs.

    This function performs validates the inputs to ensure downstream
    processing can rely on consistent types and assumptions. It verifies
    that the dataset path exists and that the optional job index contains
    only integers. The job index is normalized into a list when provided.

    Args:
        dataset (Path | str): Path to the dataset directory used for the original
            capacity-expansion run. The path must exist.
        job_index (List[int] | None): List of scenario indices to process. If
            None, all available scenarios will be processed.

    Returns:
        Tuple
            The validated dataset path and a list of scenario indices.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        TypeError: If job_index is provided and contains non-integer values.
    """
    dataset = Path(dataset)

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    if folder_output is None:
        folder_output = "./outputs/"
    folder_output = Path(folder_output)
    if not (folder_output / dataset.name).exists():
        raise FileNotFoundError(f"Results for dataset {dataset} do not exist"
            f" in the folder {folder_output}.")

    if job_index is None:
        job_index_list = None
    else:
        job_index_list = list(job_index)
        if not all(isinstance(i, int) for i in job_index_list):
            raise TypeError("job_index must be an iterable of integers")

    return dataset, folder_output, job_index_list


def load_scenarios(
    results_path: Path,
    job_index: List[int] | None,
) -> List[str]:
    """Load scenario names from simulation results.

    This function inspects the results of a previous capacity-expansion run
    and extracts the scenario names. When a job index is provided, only the 
    scenarios corresponding to those indices are returned.

    Args:
        results_path: Path to the directory containing simulation results.
        job_index: List of indices identifying which scenarios to load.
            If empty or None, all scenarios are returned.

    Returns:
        List[str]:
            A list of scenario names corresponding to the selected indices.

    Raises:
        ValueError: If no scenarios are found in the results directory.
        IndexError: If job_index contains indices that are out of range.
    """
    results = Results(results_path)
    scenarios = list(results.solution_loader.scenarios.keys())

    if not scenarios:
        raise ValueError("No scenarios found in simulation results")

    if job_index:
        scenarios = [scenarios[i] for i in job_index]

    return scenarios


def prepare_operational_dataset(
    dataset: Path,
    dataset_op: Path,
    folder_output: Path,
    scenario: str,
    scenarios_op,
) -> None:
    """Create and configure an operational-only dataset for a scenario.

    This function derives an operational dataset from the original
    capacity-expansion dataset by copying the base dataset, adding
    capacity expansion results as existing capacities, and disabling further 
    investment decisions.

    Args:
        dataset (Path): Path to the original capacity-expansion dataset.
        dataset_op (Path): Destination path for the generated operational dataset.
        folder_output (Path): Path to the directory containing capacity-expansion
            results.
        scenario (str): Name of the scenario whose results should be used in the 
            operational scenarios.
        scenarios_op (str): Name of the file containing scenario configurations
            for operational analysis. If provided, scenario analysis is enabled
            in the operational simulations.

    Side Effects:
        - Creates files and directories under `dataset_op`.
    """
    logger.info("Preparing operational dataset: %s", dataset_op)

    utils.copy_dataset(
        dataset,
        dataset_op,
        scenarios=scenarios_op,
    )

    utils.capacity_addition_2_existing_capacity(
        folder_output,
        dataset,
        dataset_op,
        scenario
    )

    utils.modify_json(
        dataset_op / "system.json",
        {
            "allow_investment": False,
            "conduct_scenario_analysis": scenarios_op is not None,
        },
    )


def run_operational_simulation(
    dataset_op: Path,
    config: Path,
    folder_output: Path,
) -> None:
    """Run an operational-only simulation.

    Executes the Zen Garden simulation using a dataset that has been
    prepared specifically for operational analysis (i.e., investment
    decisions are disabled).

    Args:
        dataset_op (Path): Path to the operational dataset.
        config (Path): Path to the simulation configuration file.
        folder_output (Path): Directory where simulation outputs will be written.
    """
    logger.info("Running operational simulation for %s", dataset_op.name)
    run(dataset=dataset_op, config=config, folder_output=folder_output)


def cleanup_dataset(dataset_op: Path, delete_data: bool) -> None:
    """Remove a generated operational dataset directory if requested.

    This helper function provides controlled cleanup of intermediate
    datasets created during operational runs.

    Args:
        dataset_op (Path): Path to the created operational dataset.
        delete_data (bool): If True, the dataset directory and all of its contents
            are permanently deleted.
    """
    if delete_data:
        logger.info("Deleting dataset: %s", dataset_op)
        shutil.rmtree(dataset_op)


def operation_scenarios(
    dataset: Path | str,
    config: Path | str = Path("./config.json"),
    folder_output: Path | str = Path("./outputs"),
    job_index: Optional[Iterable[int]] = None,
    scenarios_op: str | None = None,
    delete_data: bool = False
) -> None:
    """
    Run operational-only simulations derived from expansion results.

    This is the main orchestration function for running operational
    scenarios. For each selected scenario, it validates inputs, prepares
    an operational dataset, executes the operation simulation, and optionally
    cleans up intermediate data.

    Args:
        dataset (str | Path): Path to the original dataset used for 
            capacity-expansion runs.
        config (str | Path): Path to the simulation configuration file.
        folder_output (str | Path): Directory containing simulation outputs of 
            the capacity-planning problem. New operation results will also be 
            saved in this directory. Defaults to "./outputs/".
        job_index (list[int]): Optional iterable of scenario indices in the 
            capacity-planning problem to run. Only these scenarios will be 
            used in the operation-only simulations. If None, all scenarios are 
            processed.
        scenarios_op (str): Name of the scenario configuration for operational
            analysis.
        delete_data: If True, generated operational datasets are deleted
            after use.

    Side Effects:
        - Creates and optionally deletes dataset directories.
        - Executes simulation runs and writes output files to disk.
        - Emits log messages during execution.
    """
    dataset, folder_output, job_index_list = validate_inputs(
        dataset, folder_output, job_index)

    dataset_path = dataset.parent
    dataset_name = dataset.name
    results_path = folder_output / dataset_name

    scenarios = load_scenarios(results_path, job_index_list)

    for scenario in scenarios:
        dataset_op = dataset_path / f"{dataset_name}_{scenario}__operation"

        prepare_operational_dataset(
            dataset=dataset,
            dataset_op=dataset_op,
            folder_output=results_path,
            scenario=scenario,
            scenarios_op=scenarios_op,
        )

        run_operational_simulation(
            dataset_op=dataset_op,
            config=config,
            folder_output=folder_output,
        )

        cleanup_dataset(dataset_op, delete_data)