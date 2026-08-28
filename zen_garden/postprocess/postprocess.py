"""Class is defining the postprocessing of the results.
The class takes as inputs the optimization problem (model) and the system
configurations (system). The class contains methods to read the results and
save them in a result dictionary (resultDict).
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable, Literal, cast

import numpy as np
import pandas as pd
import pint
import xarray as xr
import yaml
from filelock import FileLock
from tables import NaturalNameWarning

if TYPE_CHECKING:
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.model.zen_model import ZenModel
    from zen_garden.preprocess.scaling import Scaling
    from zen_garden.preprocess.unit_handling import UnitHandling

logger = logging.getLogger(__name__)

# Warnings
warnings.filterwarnings("ignore", category=NaturalNameWarning)

H5_COMP_LEVEL: int = 4
H5_COMP_LIB: Literal["zlib", "lzo", "bzip2", "blosc"] = "blosc"

UNRELATED_INDEXES_FOR_UNITS = set(
    [
        "set_location",
        "set_nodes",
        "set_edges",
        "set_time_steps_operation",
        "set_time_steps_storage",
        "set_years",
    ]
)


class Postprocess:
    """Class is defining the postprocessing of the results."""

    def __init__(
        self,
        model_schema: "ModelSchema",
        unit_handling: "UnitHandling",
        zen_model: "ZenModel",
        scaling: "Scaling",
        time_steps: "TimeStepsDicts",
        optimized_time_steps: list[int],
        scenarios,
        model_name: str,
        subfolder: tuple[Path, Path] | Path,
        param_map,
    ):
        """Postprocessing of the results of the optimization.

        :param model: optimization model
        :param model_name: The name of the model used to name the output folder
        :param subfolder: The subfolder used for the results
        :param scenario_name: The name of the current scenario
        :param param_map: A dictionary mapping the parameters to the scenario names
        """
        logger.info("\n--- Postprocess results ---\n")
        # get the necessary stuff from the model
        self.model_schema = model_schema
        self.unit_handling = unit_handling
        self.zen_model = zen_model
        self.energy_system = model_schema.energy_system

        self.lp_model = zen_model.lp_model

        self.optimized_time_steps = optimized_time_steps
        self.scenarios = scenarios
        self.param_map = param_map
        self.scaling = scaling
        self.time_steps = time_steps

        # get name or directory
        self.model_name: str = model_name
        self.name_dir: Path = Path(self.config.analysis.folder_output) / self.model_name

        # deal with the subfolder
        self.subfolder = subfolder
        # here we make use of the fact that None and "" both evaluate to
        # False but any non-empty string doesn't
        if subfolder != Path(""):
            # check if mf within scenario analysis
            if isinstance(self.subfolder, tuple):
                scenario_dir = self.name_dir.joinpath(self.subfolder[0])
                os.makedirs(scenario_dir, exist_ok=True)
                mf_in_scenario_dir = self.subfolder[0].joinpath(self.subfolder[1])
                self.name_dir = self.name_dir.joinpath(mf_in_scenario_dir)
            else:
                self.name_dir = self.name_dir.joinpath(self.subfolder)
        # create the output directory
        os.makedirs(self.name_dir, exist_ok=True)

        # check if we should overwrite output
        self.overwrite = self.config.analysis.overwrite_output
        # get the compression param
        self.output_format = self.config.analysis.output_format

    def save_results(self, scenario_name: str | None):
        """Saves the results of the optimization to a folder.

        :param scenario_name: name of scenario for which results are postprocessed
        """

        # save components
        component_map: dict[str, list[Hashable]] = {}
        component_map["sets"] = self.save_sets()
        component_map["parameter"] = self.save_param()
        component_map["variable"] = self.save_var()
        component_map["dual"] = self.save_duals()
        component_map["reduced_cost"] = self.save_reduced_costs()
        self.save_component_map(component_map)

        self.save_system()
        self.save_analysis()
        self.save_scenarios()
        self.save_solver()
        self.save_unit_definitions()
        self.save_sequence_time_steps(scenario=scenario_name)
        self.save_param_map()
        if self.config.solver.run_diagnostics:
            self.save_benchmarking_data()

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def save_benchmarking_data(self):
        """Saves the benchmarking data to a json file."""
        # initialize dictionary
        benchmarking_data = dict()
        # get the benchmarking data
        benchmarking_data["objective_value"] = self.lp_model.objective.value
        if self.config.solver.name == "gurobi":
            benchmarking_data["solving_time"] = self.lp_model.solver_model.Runtime
            if "Method" in self.config.solver.solver_options:
                if self.config.solver.solver_options["Method"] == 2:
                    benchmarking_data["number_iterations"] = (
                        self.lp_model.solver_model.BarIterCount
                    )
                else:
                    benchmarking_data["number_iterations"] = (
                        self.lp_model.solver_model.IterCount
                    )
            benchmarking_data["solver_status"] = self.lp_model.solver_model.Status
            benchmarking_data["number_constraints"] = (
                self.lp_model.solver_model.NumConstrs
            )
            benchmarking_data["number_variables"] = self.lp_model.solver_model.NumVars
        elif self.config.solver.name == "highs":
            benchmarking_data["solver_status"] = (
                self.lp_model.solver_model.getModelStatus().name
            )
            benchmarking_data["solving_time"] = self.lp_model.solver_model.getRunTime()
            benchmarking_data["number_iterations"] = (
                self.lp_model.solver_model.getInfo().simplex_iteration_count
            )
            benchmarking_data["number_constraints"] = (
                self.lp_model.solver_model.getNumRow()
            )
            benchmarking_data["number_variables"] = (
                self.lp_model.solver_model.getNumCol()
            )
        else:
            logger.info(
                f"Saving benchmarking data for solver {self.config.solver.name} has "
                "not been implemented yet"
            )

        benchmarking_data["scaling_time"] = self.scaling.scaling_time
        # get numerical range
        range_lhs, range_rhs = self.scaling.print_numerics(
            0, no_scaling=False, benchmarking_output=True
        )
        benchmarking_data["numerical_range_lhs"] = range_lhs
        benchmarking_data["numerical_range_rhs"] = range_rhs
        fname = self.name_dir.joinpath("benchmarking.json")
        self._write_json_file(fname, benchmarking_data)

    def save_sets(self) -> list[Hashable]:
        """Saves the Set values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some
        other time.
        """

        series: dict[str, pd.Series] = {}
        for set in self.zen_model.sets:
            if not set.is_indexed():
                continue

            data = [",".join([str(t) for t in tpl]) for tpl in set.data.values()]
            indices_list = list(set.data.keys())
            if len(indices_list) >= 1 and isinstance(indices_list[0], tuple):
                indices = pd.MultiIndex.from_tuples(indices_list, names=[set.name])
            else:
                indices = pd.Index(data=indices_list, name=set.name)

            series[set.name] = pd.Series(data, name=set.name, index=indices)

        self._write_h5_file(self.name_dir / "sets.h5", series)
        self._write_json_file(self.name_dir / "sets_docs", self.zen_model.sets.docs)

        return list(series.keys())

    def save_param(self) -> list[Hashable]:
        """Saves the Param values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other
        time.
        """
        if not self.config.solver.save_parameters:
            logger.info("Parameters are not saved")
            return []

        parameters = xr.Dataset()
        for param in self.zen_model.parameters.docs.keys():
            if (
                self.config.solver.selected_saved_parameters
                and param not in self.config.solver.selected_saved_parameters
            ):
                continue
            # get the values
            vals = getattr(self.zen_model.parameters, param)
            # data frame
            if isinstance(vals, xr.DataArray):
                parameters[param] = vals
            # we have a scalar
            else:
                parameters[param] = xr.DataArray(data=[vals], dims=["scalar"])

        self._write_netcdf_file(self.name_dir / "parameters.nc", parameters)
        units = {
            name: pd.Series(value)
            for name, value in self.zen_model.parameters.units.items()
            if value is not None
        }
        self._write_units_to_file(self.name_dir / "parameters_units.h5", units)
        self._write_json_file(
            self.name_dir / "parameters_docs.json", self.zen_model.parameters.docs
        )

        return list(parameters.keys())

    def save_var(self) -> list[Hashable]:
        """Saves the variable values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other
        time.
        """
        self._write_netcdf_file(self.name_dir / "variables.nc", self.lp_model.solution)

        units = {
            cast(str, name): self.zen_model.variables.units[cast(str, name)]
            for name in self.lp_model.solution.keys()
        }
        units_filtered = {
            name: series
            for name, series in units.items()
            if series is not None and not series.empty
        }
        self._write_units_to_file(self.name_dir / "variables_units.h5", units_filtered)
        self._write_json_file(
            self.name_dir / "variables_docs.json", self.zen_model.variables.docs
        )
        return list(self.lp_model.solution.keys())

    def save_duals(self) -> list[Hashable]:
        """Saves the dual variable values to a h5 file."""
        if not self.config.solver.save_duals:
            logger.info("Duals are not saved")
            return []

        self._write_netcdf_file(self.name_dir / "duals.nc", self.lp_model.dual)
        self._write_json_file(
            self.name_dir / "duals_docs", self.zen_model.constraints.docs
        )
        return list(self.lp_model.dual.keys())

    def save_reduced_costs(self) -> list[Hashable]:
        """Saves the reduced cost values of variables to a h5 file."""
        if self.config.solver.name != "gurobi":
            logger.info("Reduced costs are only supported for gurobi solver")
            return []

        if not self.config.solver.save_reduced_costs:
            logger.info("Reduced costs are not saved")
            return []

        reduced_costs = xr.Dataset()
        for name in self.lp_model.variables:
            # skip variables not selected to be saved
            if (
                self.config.solver.selected_saved_reduced_costs
                and name not in self.config.solver.selected_saved_reduced_costs
            ):
                continue

            # get reduced costs from solver
            if name in self.lp_model.variables:
                arr = self.lp_model.variables[name].get_solver_attribute("RC")
            else:
                logger.warning(f"Variable {name} not found in the model")
                continue

            # rescale
            if self.config.solver.use_scaling:
                arr = self.scaling.rescale_dataarray(arr, name)

            reduced_costs[name] = arr

        self._write_netcdf_file(self.name_dir / "reduced_costs.nc", reduced_costs)
        return list(reduced_costs.keys())

    def save_component_map(self, component_map: dict[str, list[Hashable]]):
        """Saves a list of components per type."""
        self._write_json_file(self.name_dir / "component_map.json", component_map)

    def save_system(self):
        """Saves the system dict as json."""
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir
        self._write_json_file(dirname / "system.json", self.config.system.model_dump())

    def save_analysis(self):
        """Saves the analysis dict as json."""
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir
        # remove cwd path part to avoid saving the absolute path
        if os.path.isabs(self.config.analysis.dataset):
            cwd = os.getcwd()
            self.config.analysis.dataset = os.path.relpath(
                self.config.analysis.dataset, cwd
            )
            self.config.analysis.folder_output = os.path.relpath(
                self.config.analysis.folder_output, cwd
            )
        self._write_json_file(
            dirname / "analysis.json", self.config.analysis.model_dump()
        )

    def save_solver(self):
        """Saves the solver dict as json."""
        # This we only need to save once
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir

        # remove cwd path part to avoid saving the absolute path
        if os.path.isabs(self.config.solver.solver_dir):
            cwd = os.getcwd()
            self.config.solver.solver_dir = os.path.relpath(
                self.config.solver.solver_dir, cwd
            )
        # save
        self._write_json_file(dirname / "solver.json", self.config.solver.model_dump())

    def save_scenarios(self):
        """Saves the scenario dict as json."""
        # only save the scenarios at the highest level
        fname = (
            Path(self.config.analysis.folder_output)
            / self.model_name
            / "scenarios.json"
        )
        self._write_json_file(fname, self.scenarios)

    def save_unit_definitions(self):
        """Saves the user-defined units as txt."""
        if self.config.system.use_rolling_horizon:
            dirname = self.name_dir.parent
        else:
            dirname = self.name_dir

        # Only save user-defined units (skip base units like 'meter')
        all_units = self.unit_handling.ureg._units
        default_units = pint.UnitRegistry()._units
        user_units = list(set(all_units.items()).difference(default_units.items()))
        lines = list(set(unit.raw for _, unit in user_units if hasattr(unit, "raw")))  # type: ignore[attr-defined]

        self._write_txt_file(dirname / "unit_definitions.txt", "\n".join(lines))

    def save_param_map(self):
        """Saves the param_map dict as yaml."""
        if self.param_map is None:
            return

        # This we only need to save once
        if (
            self.config.system.use_rolling_horizon
            and self.config.system.conduct_scenario_analysis
        ):
            fname = self.name_dir.parent.parent.joinpath("param_map")
        elif self.subfolder != Path(""):
            fname = self.name_dir.parent.joinpath("param_map")
        else:
            fname = self.name_dir.joinpath("param_map")

        self._write_yml_file(fname.with_suffix(".yml"), self.param_map)

    def save_sequence_time_steps(self, scenario: str | None = None):
        """Saves the dict_all_sequence_time_steps dict as json.

        :param scenario: name of scenario for which results are postprocessed
        """
        # extract and save sequence time steps, we transform the arrays to lists
        dict_sequence_time_steps = self.flatten_dict(
            self.time_steps.get_sequence_time_steps_dict()
        )
        dict_sequence_time_steps["optimized_time_steps"] = self.optimized_time_steps
        dict_sequence_time_steps["time_steps_operation_duration"] = (
            self.time_steps.time_steps_operation_duration
        )
        dict_sequence_time_steps["time_steps_storage_duration"] = (
            self.time_steps.time_steps_storage_duration
        )
        dict_sequence_time_steps["time_steps_storage_level_startend_year"] = (
            self.time_steps.time_steps_storage_level_startend_year
        )
        dict_sequence_time_steps["time_steps_year2operation"] = (
            self.get_time_steps_year2operation()
        )
        dict_sequence_time_steps["time_steps_year2storage"] = (
            self.get_time_steps_year2storage()
        )

        # add the scenario name
        if scenario is not None:
            add_on = f"_{scenario}"
        else:
            add_on = ""

            # This we only need to save once
        if self.config.system.use_rolling_horizon:
            fname = self.name_dir.parent.joinpath(
                f"dict_all_sequence_time_steps{add_on}"
            )
        else:
            fname = self.name_dir.joinpath(f"dict_all_sequence_time_steps{add_on}")
        dict_formatted = {}
        for k, v in dict_sequence_time_steps.items():
            if isinstance(v, np.ndarray):
                dict_formatted[k] = v.tolist()
            elif isinstance(v, dict):
                dict_formatted[k] = {
                    str(kk): vv.tolist() if isinstance(vv, np.ndarray) else str(vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                dict_formatted[k] = v
            else:
                NotImplementedError(f"Type {type(v)} not supported for key {k}")

        fname = Path(fname).with_suffix(".json")
        self._write_json_file(fname, dict_formatted)

    def flatten_dict(self, dictionary):
        """Creates a copy of the dictionary where all numpy arrays are
        recursively flattened to lists such that it can be saved as json file.

        :param dictionary: The input dictionary
        :return: A copy of the dictionary containing lists instead of arrays
        """
        # create a copy of the dict to avoid overwrite
        out_dict = dict()

        # falten all arrays
        for k, v in dictionary.items():
            # transform the key None to 'null'
            if k is None:
                k = "null"

            # recursive call
            if isinstance(v, dict):
                out_dict[k] = self.flatten_dict(v)  # flatten the array to list
            elif isinstance(v, pd.Series):
                # Note: list(v) creates a list of np objects v.tolist() not
                out_dict[k] = v.values.tolist()
            # take as is
            else:
                out_dict[k] = v

        return out_dict

    def get_time_steps_year2operation(self):
        """Returns a HDF5-Serializable version of the
        dict_time_steps_year2operation dictionary.
        """
        assert self.time_steps.time_steps_year2operation is not None
        return {
            str(year): time_steps
            for year, time_steps in self.time_steps.time_steps_year2operation.items()
        }

    def get_time_steps_year2storage(self):
        """Returns a HDF5-Serializable version of the
        dict_time_steps_year2storage dictionary.
        """
        assert self.time_steps.time_steps_year2storage is not None
        return {
            str(year): time_steps
            for year, time_steps in self.time_steps.time_steps_year2storage.items()
        }

    def _write_units_to_file(self, f_name: Path, units_dict: dict[str, pd.Series]):
        """Writes the units dictionary to a h5 file.

        Args:
            name: Filename without extension
            units_dict: The dictionary to save
        """
        units_dict_reduced = {}
        for key, series in units_dict.items():
            levels_to_drop = cast(
                list[Hashable],
                list(UNRELATED_INDEXES_FOR_UNITS & set(series.index.names)),
            )
            if levels_to_drop and len(levels_to_drop) < len(series.index.names):
                reduced_series = series.droplevel(levels_to_drop)
                reduced_series = reduced_series[
                    ~reduced_series.index.duplicated(keep="first")
                ]
            elif levels_to_drop:
                # if there are no index levels left, we must only keep one entry
                reduced_series = series.reset_index(drop=True).loc[[0]]
            else:
                reduced_series = series
            units_dict_reduced[key] = reduced_series
        self._write_h5_file(f_name, units_dict_reduced)

    def _write_json_file(self, file_name: Path, dict: dict[str, Any]):
        """Writes the dictionary to a json file.

        :param file_name: The name of the file
        :param dict: The dictionary to save
        """
        file_name = file_name.with_suffix(".json")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".json.lock")).acquire(timeout=300):
            with open(file_name, "w+") as file:
                json.dump(dict, file, indent=2)

    def _write_yml_file(self, file_name: Path, dict: dict[str, Any]):
        """Writes the dictionary to a yml file.

        :param file_name: The name of the file
        :param dict: The dictionary to save
        """
        file_name = file_name.with_suffix(".yml")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".yml.lock")).acquire(timeout=300):
            with open(file_name, "w") as file:
                yaml.dump(dict, file)

    def _write_txt_file(self, file_name: Path, txt: str):
        """Writes the text to a txt file.

        :param file_name: The name of the file
        :param txt: The text to save
        """
        file_name = file_name.with_suffix(".txt")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".txt.lock")).acquire(timeout=300):
            with open(file_name, "w+", encoding="utf-8") as outfile:
                outfile.write(txt)

    def _write_h5_file(
        self,
        file_name: Path,
        named_series: dict[str, pd.Series],
        mode: Literal["a", "w", "r", "r+"] = "w",
    ):
        """Writes the dictionary to a hdf5 file.

        :param file_name: The name of the file
        :param named_series: The dictionary to save
        :param mode: Writting mode for python file. The two options are 'w' and
            'a'. The former create a new file while the latter will append to an
            existing file.
        """
        file_name = file_name.with_suffix(".h5")
        if file_name.exists() and mode == "w" and not self.overwrite:
            raise FileExistsError(
                "File already exists. Please set overwrite=True to overwrite the file."
            )

        with FileLock(file_name.with_suffix(".h5.lock")).acquire(timeout=300):
            with pd.HDFStore(
                file_name, mode=mode, complevel=H5_COMP_LEVEL, complib=H5_COMP_LIB
            ) as store:
                for key, series in named_series.items():
                    if not isinstance(series, pd.Series):
                        raise TypeError(
                            (
                                f"Expected a pandas Series for key '{key}', "
                                f"but got {type(series)}"
                            )
                        )
                    store.put(key, series)

    def _write_netcdf_file(self, file_name: Path, dataset: xr.Dataset):
        """Writes the dataset to a netcdf file.

        :param file_name: The name of the file
        :param dataset: The dataset to save
        """
        file_name = file_name.with_suffix(".nc")
        if file_name.exists() and not self.overwrite:
            return

        with FileLock(file_name.with_suffix(".nc.lock")).acquire(timeout=300):
            encoding = {
                var: {"zlib": True, "complevel": H5_COMP_LEVEL}
                for var in dataset.data_vars
            }
            dataset.to_netcdf(file_name, encoding=encoding)
