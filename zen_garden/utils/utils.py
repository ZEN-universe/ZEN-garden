"""Class is defining to read in the results of an Optimization problem."""

import io
import json
import logging
import os
import sys
import warnings
import zipfile
from importlib.metadata import metadata, version

import linopy as lp
import numpy as np
import pandas as pd
import requests
import xarray as xr
from ordered_set import OrderedSet

logger = logging.getLogger(__name__)


def setup_logger(level: int | str = logging.INFO):
    """Set up logger.

    :param level: logging level
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

    logger.info(f"Running ZEN-garden version: {version('zen-garden')}")


def get_inheritors(klass):
    """Get all child classes of a given class.

    :param klass: The class to get all children
    :return: All children as a set
    """
    subclasses = OrderedSet()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def download_example_dataset(dataset):
    """Downloads a dataset example to the current working directory. The function
    downloads the ZEN-garden dataset examples from the ZEN-garden Zenodo
    repository. It then extracts the dataset specified by the user and saves
    it to the current working directory. In addition, it also downloads a
    ``config.json`` file and a Jupyter notebook demonstrating how to analyze
    the results of a model.

    Args:
        dataset (str): Name of the dataset to be downloaded. The following
            options are currently available: "1_base_case",
            "2_multi_year_optimization", "3_reduced_import_availability",
            "4_multiple_time_steps_per_year", "5_reduced_import_availability_yearly",
            "6_time_series_aggregation", "7_yearly_variation", "8_myopic_foresight",
            "9_brown_field", "10_multi_scenario",
            "11_multiple_in_output_carriers_conversion", "12_yearly_interpolation",
            "13_retrofitting_and_fuel_substitution",
            "14_unit_consistency_expected_error"

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
    assert url is not None, "Could not retrieve Zenodo metadata for zen-garden."
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
            local_folder_path = os.path.join(local_example_path, filename_ending)
            if file.is_dir():
                if not os.path.exists(local_folder_path):
                    os.mkdir(os.path.join(local_example_path, filename_ending))
            else:
                local_file_path = os.path.join(local_example_path, filename_ending)
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
                local_dataset_path, "example_notebook.ipynb"
            )
            notebook = json.loads(zenodo_zip.read(file))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":  # Check only code cells
                    for i, line in enumerate(cell["source"]):
                        if "<dataset_name>" in line:
                            cell["source"][i] = line.replace("<dataset_name>", dataset)
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
            "Config.json file could not be downloaded from the dataset examples!"
        )
    if not notebook_found:
        warnings.warn(
            "Example jupyter notebook could not be downloaded from the "
            "dataset examples!",
            stacklevel=2,
        )

    # print output
    logger.info(f"Example dataset {dataset} downloaded to {local_example_path}")

    # return
    return local_example_path, os.path.join(local_dataset_path, "config.json")


# linopy helpers
# --------------


def align_like(da, other, fillna=0.0, astype=None):
    """Aligns a data array like another data array.

    :param da: The data array to align
    :param other: The data array to align to
    :param fillna: The value to fill na values with
    :param astype: The type to cast the data array to
    :return: The aligned data array
    """
    if isinstance(other, lp.Variable):
        other = other.lower
    elif isinstance(other, lp.LinearExpression):
        other = other.const
    elif isinstance(other, xr.DataArray):
        other = other
    else:
        raise TypeError(
            "other must be a Variable, LinearExpression or DataArray, "
            f"not {type(other)}"
        )
    da = xr.align(da, other, join="right")[0]
    da = da.broadcast_like(other)
    if fillna is not None:
        da = da.fillna(fillna)
    if astype is not None:
        da = da.astype(astype)
    return da


def linexpr_from_tuple_np(tuples, coords, model):
    """Transforms tuples of (coeff, var) into a linopy linear expression.

    Uses numpy broadcasting.

    :param tuples: Tuple of (coeff, var)
    :param coords: The coordinates of the final linear expression
    :param model: The model to which the linear expression belongs
    :return: A linear expression
    """
    # get actual coords
    if not isinstance(coords, xr.core.dataarray.DataArrayCoordinates):
        coords = xr.DataArray(coords=coords).coords

    # numpy stack everything
    coefficients = []
    variables = []
    for coeff, var in tuples:
        var = var.labels.data
        if isinstance(coeff, (float, int)):
            coeff = np.full(var.shape, 1.0 * coeff)
        coefficients.append(coeff)
        variables.append(var)

    # to linear expression
    variables = xr.DataArray(
        np.stack(variables, axis=0), coords=coords, dims=["_term", *coords]
    )
    coefficients = xr.DataArray(
        np.stack(coefficients, axis=0), coords=coords, dims=["_term", *coords]
    )
    xr_ds = xr.Dataset({"coeffs": coefficients, "vars": variables}).transpose(
        ..., "_term"
    )

    return lp.LinearExpression(xr_ds, model)


def xr_like(fill_value, dtype, other, dims):
    """Create an xarray with fill value and dtype like the other object.

    Only contains the given dimensions.

    :param fill_value: The value to fill the data with
    :param dtype: dtype of the data
    :param other: The other object to use as base
    :param dims: The dimensions to use
    :return: An object like the other object but only containing the given dimensions
    """
    # get the coords
    coords = {}
    for dim in dims:
        coords[dim] = other.coords[dim]

    # create the data array
    da = xr.DataArray(
        np.full([len(other.coords[dim]) for dim in dims], fill_value, dtype=dtype),
        coords=coords,
        dims=dims,
    )

    # return
    return da


def reformat_slicing_index(index, component) -> tuple[str]:
    """Reformats the slicing index to a tuple of strings that is readable by pytables
    :param index: slicing index of the resulting dataframe
    :param component: component for which the index is reformatted
    :return: reformatted index.
    """
    if index is None:
        return tuple()
    index_names = component.index_names
    if isinstance(index, str) or isinstance(index, float) or isinstance(index, int):
        index_name = index_names[0]
        ref_index = (f"'{index_name}' == '{index}'",)
        if len(index_names) == 1:
            ref_index = (f"index == '{index}'",)
    elif isinstance(index, list):
        index_name = index_names[0]
        ref_index = (f"'{index_name}' in {index}",)
    elif isinstance(index, dict):
        ref_index = []
        for key, value in index.items():
            if key not in index_names:
                warnings.warn(
                    f"Invalid index name '{key}' in index. Skipping.",
                    Warning,
                    stacklevel=2,
                )
                continue
            if isinstance(value, list):
                ref_index.append(f"'{key}' in {value}")
            else:
                ref_index.append(f"'{key}' == '{value}'")
        ref_index = tuple(ref_index)
    elif isinstance(index, tuple):
        ref_index = []
        if len(index) > len(index_names):
            warnings.warn(
                f"Index length {len(index)} is longer than the number of index "
                f"dimensions {len(index_names)}. Check selected index.",
                Warning,
                stacklevel=2,
            )
        for i, index_name in enumerate(index_names):
            if i >= len(index):
                break
            if index[i] is None:
                continue
            elif isinstance(index[i], list):
                ref_index.append(f"'{index_name}' in {index[i]}")
            else:
                ref_index.append(f"'{index_name}' == '{index[i]}'")
        ref_index = tuple(ref_index)
    else:
        warnings.warn(
            f"Invalid index type {type(index)}. Skipping.", Warning, stacklevel=2
        )
        ref_index = tuple()

    return ref_index


def slice_df_by_index(df, index_tuple) -> dict:
    """Recreate the slicing index from a tuple of strings and slice the dataframe.

    :param df: dataframe to be sliced
    :param index_tuple: tuple of strings representing the slicing index
    :return: sliced dataframe.
    """
    index = {}
    for index_str in index_tuple:
        if " in " in index_str:
            key, value_str = index_str.split(" in ")
            key = key.strip("'")
            value = eval(value_str)
        elif " == " in index_str:
            key, value_str = index_str.split(" == ")
            key = key.strip("'")
            value = eval(value_str)
        else:
            continue
        index[key] = value
    for key in index:
        if key in df.index.names:
            if isinstance(index[key], list):
                df = df.loc[df.index.get_level_values(key).isin(index[key])]
            elif index[key] in df.index.get_level_values(key):
                df = df.xs(index[key], level=key, drop_level=False)
            else:
                df = pd.DataFrame(
                    columns=df.columns
                )  # return empty dataframe if value not in index
    return df


def get_label_position(obj, label: int):
    """Get dict of index and coordinate for variable or constraint labels."""
    name_element = obj.get_name_by_label(int(label))
    element = obj[name_element]
    if element.ndim > 0:
        selection = element[np.where(element.labels == label)]
        mapping = (
            name_element,
            {k: v.values[0] for k, v in selection.indexes.variables.items()},
        )
    else:
        mapping = (name_element, {})
    return mapping
