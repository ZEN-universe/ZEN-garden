import argparse
import sys
from zen_garden.utils import download_example_dataset

def create_zen_example_cli():
    """ 
    Entry point for the `zen-example` command-line interface.

    Creates a command line interface for downloading the dataset examples. 
    The function parses a single required argument ``--dataset`` that 
    specifies the name of the dataset to be downloaded. It then invokes
    the function function ``download_example_dataset`` with that argument.

    The ``[project.scripts]`` section of the pyproject.toml declares that 
    this function will be called whenever a user enters ``zen-example`` into
    the command prompt. This function is therefore creates the ``zen-example`` 
    command line entry point.

    Examples:
        Basic usage in a command line prompt:

        >>> zen-example --dataset="1_base_case"

    """
    # parse the args
    description = "Downloads an example dataset for ZEN-garden to the current" \
        "working directory"

    parser = argparse.ArgumentParser(
        description=description,
        add_help=True,
        usage="usage: zen-example [--dataset DATASET]")

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Name of the dataset to download, e.g. '1_base_case'")

    args = parser.parse_args(sys.argv[1:])

    # download the example
    download_example_dataset(args.dataset)