"""
This module contains code the coding for downloading the ZEN-garden 
dataset examples. It defines a command line interface as well as a python 
function for this purpose. 
"""

import argparse
import sys
import json
import warnings
import os
import sys
import requests
from importlib.metadata import metadata
import zipfile
import io


def download_example_dataset(dataset):
    """ 
    Downloads a dataset example to the current working directory. The function 
    downloads the ZEN-garden dataset examples from the ZEN-garden Zenodo 
    repository. It then extracts the dataset specified by the user and saves
    it to the current working directory. In addition, it also downloads a 
    ``config.json`` file and a Jupyter notebook demonstrating how to analyze
    the results of a model. 

    Args:
        dataset (str): Name of the dataset to be downloaded. The following
            options are currently available: "1_base_case", 
            "2_multi_year_optimization", "3_reduced_import_availability", 
            "4_PWA_nonlinear_capex", "5_multiple_time_steps_per_year",
            "6_reduced_import_availability_yearly", "7_time_series_aggregation",
            "8_yearly_variation", "9_myopic_foresight", "10_brown_field",
            "11_multi_scenario", "12_multiple_in_output_carriers_conversion",
            "13_yearly_interpolation", "14_retrofitting_and_fuel_substitution",
            "15_unit_consistency_expected_error"

    Returns:
        tuple: 
            str: The local path of the copied example
            str: The local path of the copied config.json

    Raises:
        FileNotFoundError: If either the dataset or the config file could not 
            be found in the Zenodo repository. 

    Examples:
        Basic usage example:

        >>> from zen_garden.dataset_examples import download_dataset_example
        >>> download_dataset_example("1_base_case")

    """

    # retrieve Zenodo metadata
    url = metadata("zen_garden").get_all("Project-URL")
    url = [u.split(", ")[1] for u in url if u.split(", ")[0] == "Zenodo"][0]

    # fetch Zenodo metadata
    zenodo_meta = requests.get(url, allow_redirects=True)
    zenodo_meta.raise_for_status()
    zenodo_data = zenodo_meta.json()
    zenodo_zip_url = zenodo_data["files"][0]["links"]["self"]

    # download ZIP file from Zenodo
    zenodo_zip = requests.get(zenodo_zip_url)
    zenodo_zip = zipfile.ZipFile(io.BytesIO(zenodo_zip.content))

    # define relevant paths
    base_path = zenodo_zip.filelist[0].filename
    example_path = f"{base_path}docs/dataset_examples/{dataset}/"
    config_path = f"{base_path}docs/dataset_examples/config.json"
    notebook_path = f"{base_path}docs/dataset_examples/example_notebook.ipynb"

    # create local directories
    local_dataset_path = os.getcwd()
    if not os.path.exists(local_dataset_path):
        os.mkdir(local_dataset_path)
    local_example_path = os.path.join(local_dataset_path, dataset)
    if not os.path.exists(local_example_path):
        os.mkdir(local_example_path)

    # initialize flags for extracting files
    example_found = False
    config_found = False
    notebook_found = False

    # search for example within ZIP file
    for file in zenodo_zip.filelist:

        # download all files in dataset example
        if file.filename.startswith(example_path):
            filename_ending = file.filename.split(example_path)[1]
            local_folder_path = os.path.join(
                local_example_path, filename_ending)
            if file.is_dir():
                if not os.path.exists(local_folder_path):
                    os.mkdir(os.path.join(local_example_path, filename_ending))
            else:
                local_file_path = os.path.join(
                    local_example_path, filename_ending)
                with open(local_file_path, "wb") as f:
                    f.write(zenodo_zip.read(file))
            example_found = True

        # download config.json
        elif file.filename == config_path:
            with open(os.path.join(local_dataset_path, "config.json"), "wb") as f:
                f.write(zenodo_zip.read(file))
            config_found = True

        # download jupyter notebook
        elif file.filename == notebook_path:
            notebook_path_local = os.path.join(
                local_dataset_path, "example_notebook.ipynb")
            notebook = json.loads(zenodo_zip.read(file))
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':  # Check only code cells
                    for i, line in enumerate(cell['source']):
                        if "<dataset_name>" in line:
                            cell['source'][i] = line.replace(
                                "<dataset_name>", dataset)
            with open(notebook_path_local, "w") as f:
                json.dump(notebook, f)
            notebook_found = True

    # display status, errors, and warnings
    if not example_found:
        raise FileNotFoundError(
            f"Example {dataset} could not be found in the dataset examples!"
        )
    if not config_found:
        raise FileNotFoundError(
            "Config.json file could not be downloaded from the dataset "
            "examples!"
        )
    if not notebook_found:
        warnings.warn(
            "Example jupyter notebook could not be downloaded from the "
            "dataset examples!")

    # print output
    print(f"Example dataset {dataset} downloaded to {local_example_path}")

    # return
    return local_example_path, os.path.join(local_dataset_path, "config.json")


def cli_download_example_dataset():
    """ 
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
