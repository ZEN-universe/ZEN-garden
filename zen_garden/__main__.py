"""
Runs the main function of ZEN-Garden.
Compilation  of the optimization problem.
"""
from .runner import run
from .cli.zen_garden_cli import create_zen_garden_cli
import warnings

def run_module(config = "./config.py", dataset = None, 
               folder_output = None, job_index = None):
    """
    Deprecated wrapper for ``zen_garden.runner.run()``.

    This function mirrors the behavior of
    ``zen_garden.runner.run()`` and exists solely for backward
    compatibility with older versions of ZEN_garden, where
    ``zen_garden.__main__.run_module()`` served as the primary entry point.

    This function is deprecated and will be removed in ZEN-garden v3.0.0.
    Users should migrate to ``zen_garden.runner.run()``.

    Args:
        config (str): Path to the configuration file (e.g. ``config.json``).
            If the file is located in the current working directory, the
            filename alone may be specified. Defaults to ``"./config.py"``.
        dataset (str): Path to the folder containing the input dataset
            (e.g. ``"./1_base_case"``). If located in the current working
            directory, the folder name alone may be used. Defaults to the
            ``dataset`` value specified in the configuration file.
        folder_output (str): Path to the folder where outputs will be saved.
            Defaults to ``"./outputs"``.
        job_index (list[int] | None): Indices of jobs (scenarios) to run.
            For example, ``job_index=[1]`` runs only the first scenario.
            Defaults to ``None`` (run all jobs).

    Raises:
        DeprecationWarning: This function will be removed in ZEN-garden v3.0.0.

    Returns:
        OptimizationSetup: The fully set up and solved optimization problem.

    See Also:
        zen_garden.runner.run: Replacement function.

    Examples:
        >>> from zen_garden import run, copy_example_dataset
        >>> download_example_dataset("1_base_case")
        >>> run("1_base_case")
    """


    # throw deprecation warning
    warnings.warn(
        "zen_garden.__main__.run_module() is deprecated and will be removed " \
        "in ZEN-garden v3.0.0. Please use the new version " \
        "zen_garden.runner.run() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # run new function
    return run(
        config = config, 
        dataset = dataset, 
        folder_output=folder_output,
        job_index=job_index
        )

if __name__ == "__main__":

    create_zen_garden_cli()
