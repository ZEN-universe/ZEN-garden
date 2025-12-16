import argparse
from zen_garden.runner import run
import os

# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ZEN-garden.

    This function defines all supported command-line options for running
    ZEN-garden. The parser handles configuration file selection, dataset and
    output directory overrides, and job indexing for array or batch execution.

    Command Line Flags:
        --config (str, optional):
            Path to a Python or JSON configuration file. If not provided, the
            configuration is read from the current working directory. Defaults
            to "./config.json".

        --dataset (str, optional):
            Path to the dataset directory. Overrides
            ``config.analysis.dataset`` in the configuration file.

        --folder_output (str, optional):
            Path to the output directory. Overrides output-related settings in
            the configuration file. If not specified, output is written to the
            current working directory.

        --job_index (str, optional):
            Comma-separated list of scenario or job indices to execute. If not
            provided, the value is read from the environment variable specified
            by ``--job_index_var``.

        --job_index_var (str, optional):
            Name of the environment variable containing the job index.
            Defaults to ``SLURM_ARRAY_TASK_ID``.

    Returns:
        argparse.ArgumentParser: An argument parser configured for the
        ZEN-Garden command-line interface.

    """
    description = (
        "Run ZEN-Garden with a given config file. By default, the config file "
        "is read from the current working directory. You may specify a config "
        "file with --config. Output is always written to the current working "
        "directory unless overridden.")

    parser = argparse.ArgumentParser(description=description,
                                     add_help=True,
                                     usage="zen_garden [options]")

    parser.add_argument("--config",
                        type=str,
                        required=False,
                        default="./config.json",
                        help="Path to a Python or JSON config file.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        help="Path to the dataset directory. Overrides config.analysis.dataset."
    )
    parser.add_argument(
        "--folder_output",
        type=str,
        required=False,
        default=None,
        help=
        "Path to the output directory. Overrides output settings in config.")
    parser.add_argument(
        "--job_index",
        type=str,
        required=False,
        default=None,
        help="Comma-separated list of scenario indices. If omitted, the "
        "environment variable specified by --job_index_var is used.")
    parser.add_argument("--job_index_var",
                        type=str,
                        required=False,
                        default="SLURM_ARRAY_TASK_ID",
                        help="Environment variable for job index.")

    return parser

def resolve_job_index(job_index:str, job_index_var:str) -> list[int]:
    """
    Resolves the job index when running ZEN-garden from the command line.

    If the job index is directly specified using the ``job_index`` command-line
    flag, those values are used. Otherwise, the job index is extracted from the
    environment variable specified by the ``job_index_var`` command-line flag.
    If neither is defined, ``None`` is returned.

    Args:
        job_index (str): Value of the ``job_index`` command-line flag provided
            by the user.
        job_index_var (str): Value of the ``job_index_var`` command-line flag
            provided by the user.

    Returns:
        list[int] | None: List of job indices to run in the current instance of
            ZEN-garden, or ``None`` if no job index is specified.

    """
    if job_index:
        return [int(i) for i in job_index.split(",")]
    elif ((env_value := os.environ.get(job_index_var)) is not None):
        return [int(env_value)]
    else:
        return None

def create_zen_garden_cli():
    """
    Entry point for the `zen-garden` command-line interface.
    
    This function creates the command-line interface for running ZEN-garden.
    It first sets up an argument parser; extracts the job index (either from
    the input flax directly or from an environment variable), and then
    calls the ``zen_garden.run()`` function.

    The ``[project.scripts]`` section of the pyproject.toml declares that 
    this function will be called whenever a user enters ``zen-garden`` into
    the command prompt.

    Command Line Flags:
        --config (str, optional):
            Path to a Python or JSON configuration file. If not provided, the
            configuration is read from the current working directory.

        --dataset (str, optional):
            Path to the dataset directory. Overrides
            ``config.analysis.dataset`` in the configuration file.

        --folder_output (str, optional):
            Path to the output directory. Overrides output-related settings in
            the configuration file. If not specified, output is written to the
            current working directory.

        --job_index (str, optional):
            Comma-separated list of scenario or job indices to execute. If not
            provided, the value is read from the environment variable specified
            by ``--job_index_var``.

        --job_index_var (str, optional):
            Name of the environment variable containing the job index.
            Defaults to ``SLURM_ARRAY_TASK_ID``.
                   
    Returns:
        None

    Examples:
        Basic usage in a command line prompt:

        >>> zen-garden --config=".\\config.json" --dataset="1_base_case"

    """    
    # parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    ### get the job index
    job_index = resolve_job_index(args.job_index, args.job_index_var)

    run(
        config=args.config,
        dataset=args.dataset,
        folder_output=args.folder_output,
        job_index=job_index,
    )
