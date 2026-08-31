"""Contains the implementation of a SolutionLoader that reads the solution."""

import itertools
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np
import pandas as pd
import xarray as xr
from pint import UnitRegistry
from xarray.backends.netCDF4_ import NetCDF4DataStore

from zen_garden.config import Analysis, Solver, System
from zen_garden.postprocess.results.component_map import ComponentMap
from zen_garden.postprocess.results.component_type import ComponentType
from zen_garden.postprocess.results.timestep_map import TimestepMap
from zen_garden.postprocess.results.timestep_type import TimestepType

logger = logging.getLogger(__name__)

VERSION_MAP = {1: "2.0.14", 2: "2.2.15", 3: "2.9.2", 4: "3.0.0"}

OBJECTIVE_FUNCTION_MAP = {
    "total_cost": "net_present_cost",
    "total_carbon_emissions": "carbon_emissions_annual",
}

DOCS_FILENAME_MAP = {
    ComponentType.sets: "sets_docs.json",
    ComponentType.parameter: "parameters_docs.json",
    ComponentType.variable: "variables_docs.json",
    ComponentType.dual: "duals_docs.json",
}

T = TypeVar("T")
FrameOrSeries = TypeVar("FrameOrSeries", pd.DataFrame, pd.Series)

IndexValue = str | int | float
IndexElement = IndexValue | list[IndexValue] | None
Index = IndexElement | tuple[IndexElement, ...] | dict[str, IndexElement]


def _format_query_expr(raw_dim: str, values: IndexElement) -> str | None:
    """Builds the boolean expression that selects ``values`` along
    ``raw_dim`` for ``DataArray.query``. Returns ``None`` if there is
    nothing to filter (``values`` is ``None`` or an empty list).
    """
    if values is None:
        return None
    if isinstance(values, list):
        if not values:
            return None
        return " | ".join(f"({raw_dim} == {v!r})" for v in values)
    return f"{raw_dim} == {values!r}"


def _build_query(
    raw_dims: tuple[str, ...],
    header_map: dict[str, str],
    index: Index | None,
) -> dict[str, str]:
    """Converts a user-facing ``index`` filter into the raw dimension-name
    -> boolean-expression dict expected by ``DataArray.query``.

    ``index`` can be:

    - a single value (str/int/float): filters the first dimension.
    - a list of values: filters the first dimension to any of these values.
    - a tuple, positional across the component's dimensions (in the order
      returned by ``get_index_names``): each entry is a single value, a
      list of values, or ``None`` to leave that dimension unfiltered.
    - a dict, keyed by either the friendly name (e.g. ``"technology"``,
      ``"node"``, from :class:`HeaderDataInputs`) or the internal
      dimension name (e.g. ``"set_technologies"``): values as above.
      Order does not matter.

    :param raw_dims: The component's dimensions, in their internal order
        (as returned by ``DataArray.dims``/``get_index_names``).
    :param header_map: Maps internal dimension names to friendly names,
        i.e. ``self.analysis.header_data_inputs`` as a dict.
    :param index: The user-facing index filter, or ``None``.
    :return: A dict mapping raw dimension names to boolean expressions,
        suitable for ``DataArray.query``.
    """
    if index is None:
        return {}

    if isinstance(index, dict):
        friendly_to_raw: dict[str, list[str]] = {}
        for raw_dim in raw_dims:
            friendly_to_raw.setdefault(header_map.get(raw_dim, raw_dim), []).append(
                raw_dim
            )

        query: dict[str, str] = {}
        for key, values in index.items():
            if key in raw_dims:
                raw_dim = key
            elif key in friendly_to_raw and len(friendly_to_raw[key]) == 1:
                raw_dim = friendly_to_raw[key][0]
            elif key in friendly_to_raw:
                raise ValueError(
                    f"'{key}' matches multiple dimensions of this component "
                    f"({friendly_to_raw[key]}). Use the internal dimension "
                    f"name directly, e.g. "
                    f"index={{'{friendly_to_raw[key][0]}': ...}}."
                )
            else:
                raise ValueError(
                    f"'{key}' is not a dimension of this component. "
                    f"Available: {[header_map.get(d, d) for d in raw_dims]} "
                    f"(internal names: {list(raw_dims)})."
                )
            expr = _format_query_expr(raw_dim, values)
            if expr is not None:
                query[raw_dim] = expr
        return query

    if isinstance(index, tuple):
        if len(index) > len(raw_dims):
            raise ValueError(
                f"Index tuple has {len(index)} entries, but this component "
                f"only has {len(raw_dims)} dimensions: {list(raw_dims)}."
            )
        query = {}
        for raw_dim, values in zip(raw_dims, index, strict=False):
            expr = _format_query_expr(raw_dim, values)
            if expr is not None:
                query[raw_dim] = expr
        return query

    if not isinstance(index, (str, int, float, list)):
        raise TypeError(
            f"index must be a string, number, list, tuple, dict, or None. "
            f"Got {type(index)}: {index!r}."
        )

    # a bare value or list of values filters the first dimension
    expr = _format_query_expr(raw_dims[0], index)
    return {raw_dims[0]: expr} if expr is not None else {}


def _open_component_variable(file_path: Path, component_name: str) -> xr.DataArray:
    """Opens a single netCDF variable and its dimension coordinates.

    A ``variables.nc``/``parameters.nc``/``duals.nc`` file bundles every
    component of a scenario. ``xr.open_dataset`` builds a metadata wrapper
    (chunking, compression filters, attrs) for every variable in the file,
    which costs ~15-25ms per variable regardless of that variable's size.
    With dozens of components in one file, that dominates read time
    independent of how much data is actually being read. Opening only the
    requested variable and its dimension coordinates via the lower-level
    netCDF4 store avoids that per-file cost.

    :param file_path: Path to the netCDF file.
    :param component_name: The name of the component/variable to open.
    :return: A lazy DataArray for the requested component.
    """
    store = NetCDF4DataStore.open(str(file_path), mode="r")
    try:
        raw_var = store.ds.variables[component_name]
        var = store.open_store_variable(component_name, raw_var)

        coord_vars = {}
        for dim in var.dims:
            if dim != component_name and dim in store.ds.variables:
                coord_vars[dim] = store.open_store_variable(
                    dim, store.ds.variables[dim]
                )

        ds = xr.Dataset({component_name: var, **coord_vars})
        return ds[component_name]
    finally:
        store.close()


class Scenario:
    def __init__(self, path: Path, name: str, base_scenario: str) -> None:
        """Implementation of the scenario. In this solution version, the analysis and
        system configs are stored as JSONs for each of the scenario in the
        corresponding folder.

        :param path: The path to the scenario folder.
        :param name: The name of the scenario.
        :param base_scenario: The name of the base scenario.
        """
        self.name: str = name
        self.base_name: str = base_scenario
        self._path: Path = path

        self._analysis: Analysis | None = None
        self._system: System | None = None
        self._solver: Solver | None = None
        self._benchmarking: dict[str, Any] | None = None
        self._ureg: UnitRegistry | None = (  # pyright:ignore[reportMissingTypeArgument]
            None
        )
        self._component_map: ComponentMap | None = None
        self._time_steps: TimestepMap | None = None

    @property
    def analysis(self) -> Analysis:
        """Returns the analysis config information of the scenario."""
        if self._analysis is None:
            self._analysis = self._read_json_file(
                self.path / "analysis.json", Analysis, Analysis.model_construct
            )
        return self._analysis

    @property
    def solver(self) -> Solver:
        """Returns the solver config information of the scenario."""
        if self._solver is None:
            self._solver = self._read_json_file(
                self.path / "solver.json", Solver, Solver.model_construct
            )
        return self._solver

    @property
    def system(self) -> System:
        """Returns the system config information of the scenario."""
        if self._system is None:
            self._system = self._read_json_file(
                self.path / "system.json", System, System.model_construct
            )
        return self._system

    @property
    def benchmarking(self) -> dict[str, Any]:
        """Returns the benchmarking information of the scenario."""
        if self._benchmarking is None:
            self._benchmarking = self._read_json_file(
                self.path / "benchmarking.json", dict
            )
        return self._benchmarking

    @property
    def component_map(self) -> ComponentMap:
        """Returns the component map of the scenario."""
        if self._component_map is None:
            self._component_map = self._read_json_file(
                self.component_path / "component_map.json", ComponentMap
            )
        return self._component_map

    @property
    def time_steps(self) -> TimestepMap:
        """Returns the time steps of the scenario."""
        if self._time_steps is None:
            time_steps_file_name = list(
                self.path.glob("dict_all_sequence_time_steps*.json")
            )
            assert len(time_steps_file_name) == 1, (
                f"Expected exactly one time steps "
                f"file, found {len(time_steps_file_name)}"
            )
            self._time_steps = self._read_json_file(
                time_steps_file_name[0], TimestepMap
            )
        return self._time_steps

    @property
    def path(self) -> Path:
        """Path to the folder containing the scenario files."""
        return self._path

    @property
    def component_path(self) -> Path:
        """Path to the folder containing the component files.
        If the solution has rolling horizon enabled,
        the component files are stored in a subfolder named "MF_<number>".
        """
        if self.has_rh:
            mf_folder = next(
                s
                for s in sorted(self.path.iterdir())
                if s.is_dir() and s.name.startswith("MF_")
            )
            return self._path / mf_folder
        return self._path

    @property
    def has_rh(self) -> bool:
        """Returns True if the solution has rolling horizon enabled."""
        return self.system.use_rolling_horizon

    @property
    def components(self) -> list[str]:
        """Returns a list of all components in the scenario."""
        return self.component_map.all_components

    @property
    def ureg(self) -> UnitRegistry:
        """Returns the unit registry for the scenario."""
        if self._ureg is None:
            self._ureg = self._read_ureg()
        return self._ureg

    @property
    def output_version(self) -> int:
        """Returns the output version of the scenario."""
        if (
            "output_version" not in self.analysis.__pydantic_fields_set__
            or self.analysis.output_version is None
        ):
            # fallback for older solutions written before this field existed
            # (``model_construct`` still fills the schema default, so an absent
            # field cannot be detected from its value alone).
            return self._solution_version()
        return self.analysis.output_version

    def get_values(
        self,
        component_name: str,
        index: Index | None = None,
        keep_raw: bool = False,
        rename_index: bool = True,
    ):
        """Get the values for a given component filtered by the index.

        :param component_name: The name of the component.
        :param index: The index used to slice the series unless the component is a set.
        :param rename_index: Whether to rename the index of the series.
        :return: The pandas series object.
        """
        if self.has_rh:
            return self.get_values_of_rolling_horizon(
                component_name, index, keep_raw, rename_index
            )
        else:
            return self.get_raw_values(component_name, index, rename_index)

    def get_raw_values(
        self,
        component_name: str,
        index: Index | None = None,
        rename_index: bool = True,
        mf_folder: str | None = None,
    ) -> pd.Series:
        """Get the values for a given component filtered by the index.

        :param component_name: The name of the component.
        :param index: The index used to slice the series unless the component is a set.
        :param subfolder: Optional subfolder name.
        :param rename_index: Whether to rename the index of the series.
        :return: The pandas series object.
        """
        component_type = self.component_map.find_type(component_name)
        sub_folder: Path = (
            Path(".")
            if mf_folder is None or not self.has_rh
            else Path("..") / mf_folder
        )
        file_path = self.component_path / sub_folder / component_type.get_file_name()

        if component_type is ComponentType.sets:
            series = pd.read_hdf(file_path, component_name)
            assert isinstance(series, pd.Series), (
                f"Component {component_name} is not a series, but a {type(series)}. "
                f"Please check the component_map.json file."
            )
            return series

        # da = xr.open_dataset(file_path)[component_name]
        da = _open_component_variable(file_path, component_name)
        header_data_inputs = self.analysis.header_data_inputs
        header_map = (
            header_data_inputs
            if isinstance(header_data_inputs, dict)
            else header_data_inputs.model_dump()
        )
        raw_dims = tuple(str(dim) for dim in da.dims)
        query = _build_query(raw_dims, header_map, index)
        ans = da.query(query).to_series().dropna()
        if rename_index:
            ans = self._rename_index(ans)
        return ans

    def get_values_of_rolling_horizon(
        self,
        component_name: str,
        index: Index | None,
        keep_raw: bool = False,
        rename_index: bool = True,
    ) -> pd.Series:
        """Get values for a system that uses rolling horizon.

        :param component_name: The name of the component.
        :param index: The index used to slice the series unless the component is a set.
        :param rename_index: Whether to rename the index of the series.
        """
        if not self.has_rh:
            raise ValueError(
                f"Scenario {self.name} does not have rolling horizon enabled."
            )

        # If solution has rolling horizon, load the values
        # for all the foresight steps and combine them.
        subfolder_names = [
            p.name
            for p in self.path.iterdir()
            if p.is_dir() and p.name.startswith("MF_")
        ]

        series_dict: dict[int, pd.Series] = {}
        for subfolder_name in subfolder_names:
            sf_stripped = subfolder_name.replace("MF_", "")
            mf_idx: int | str
            if not sf_stripped.isnumeric():
                raise ValueError(
                    (
                        f"Subfolder name {subfolder_name} is not in the expected "
                        f"format 'MF_<number>' or 'MF_<number>_<description>'."
                    )
                )
            mf_idx = int(subfolder_name.replace("MF_", ""))
            series = self.get_raw_values(
                component_name,
                index,
                mf_folder=subfolder_name,
                rename_index=rename_index,
            )
            assert isinstance(mf_idx, int)
            series_dict[mf_idx] = series

        timestep_column, timestep_type = TimestepType.from_index_names(
            [str(name) for name in next(iter(series_dict.values())).index.names]
        )
        if keep_raw:
            return self._concatenate_raw_series(series_dict)
        return self._combine_dataseries(series_dict, timestep_column, timestep_type)

    def get_full_ts(
        self,
        component_name: str,
        year: int | None = None,
        discount_to_first_step: bool = True,
        keep_raw: bool = False,
        index: Index | None = None,
    ) -> pd.DataFrame:
        """Calculates the full timeseries per scenario.

        Args:
            scenario: The scenario for with the component should be extracted
                (only if needed)
            component: Component for the Series
            discount_to_first_step: apply annuity to first year of interval or
                entire interval
            year: year of which full time series is selected
            keep_raw: Keep the raw values of the rolling horizon optimization
            index: slicing index of the resulting dataframe

        Returns:
            Full timeseries
        """
        component_type = self.component_map.find_type(component_name)
        values = self.get_values(
            component_name, index, keep_raw=keep_raw, rename_index=False
        )
        timestep_column, timestep_type = TimestepType.from_index_names(
            [str(name) for name in values.index.names]
        )

        if timestep_type is None:
            raise ValueError(f"Component {component_name} has no timestep type.")

        sequence_timesteps = self._get_sequence_time_steps(timestep_type)
        if year is None:
            years = [i for i in range(0, self.system.optimized_years)]
        else:
            year = self._convert_year2ts(year)
            years = [year]

        # slice index with time steps of year, unless the caller already
        # filtered the timestep dimension explicitly via `index`
        select_year_time_steps = False
        time_steps: set[int] = set()
        index_has_timestep_filter = isinstance(index, dict) and any(
            str(timestep_type.value) in str(key) for key in index
        )
        if timestep_type in [TimestepType.operational, TimestepType.storage] and (
            index is None or not index_has_timestep_filter
        ):
            assert timestep_column is not None
            time_steps = self._get_timesteps_of_years(timestep_type, tuple(years))
            select_year_time_steps = True

        if isinstance(values.index, pd.MultiIndex):
            values = values.unstack(timestep_column)

        if timestep_type is TimestepType.yearly:
            if component_type is ComponentType.dual:
                annuity = self._annuity(discount_to_first_step)
                ans = values / annuity
            else:
                ans = values

            years_list = (
                [y for y in years if y in ans.columns]
                if isinstance(ans, pd.DataFrame)
                else [y for y in years if y in ans.index]
            )
            if years_list and isinstance(ans, pd.DataFrame):
                ans = ans.loc[:, years]
            elif years_list:
                ans = ans.loc[years]
            ans = ans.sort_index()
            ans = self._convert_ts2year(ans)
            ans = self._rename_index(ans)
            return ans

        if component_type is ComponentType.dual:
            timestep_duration = self._get_timestep_duration(timestep_type)
            annuity = self._annuity()
            values = values.div(timestep_duration, axis=1)

            for year_temp in annuity.index:
                time_steps_year = list(
                    self._get_timesteps_of_years(timestep_type, (year_temp,))
                )
                values[time_steps_year] = values[time_steps_year] / annuity[year_temp]

        # try:
        if timestep_type is TimestepType.operational:
            if select_year_time_steps:
                sequence_timesteps = sequence_timesteps[
                    sequence_timesteps.isin(time_steps)
                ]
            try:
                output_df = values[sequence_timesteps]
            except KeyError:
                output_df = values
        elif timestep_type is TimestepType.storage:
            # for storage components, the last timestep is the final state,
            # linear interpolation is used
            if isinstance(values, pd.Series):
                values = values.to_frame()
            last_occurrences = sequence_timesteps.drop_duplicates(keep="last")
            first_occurrences = sequence_timesteps.drop_duplicates(keep="first")
            last_occurrences = pd.Series(
                last_occurrences.index, index=last_occurrences.values
            )
            first_occurrences = pd.Series(
                first_occurrences.index, index=first_occurrences.values
            )
            last_occurrences = last_occurrences[
                last_occurrences.index.intersection(values.columns)
            ]
            output_df = values[last_occurrences.index].rename(last_occurrences, axis=1)

            # fill missing ts with nan
            time_steps_start_end = self._get_time_steps_storage_level_startend_year()
            time_steps_start_end = {
                k: v
                for k, v in time_steps_start_end.items()
                if k in first_occurrences and v in last_occurrences
            }
            for tstart, tend in time_steps_start_end.items():
                tstart_reconstructed = first_occurrences[tstart]
                _output_df_recon = output_df.iloc[0][tstart:]
                first_valid_timestep = _output_df_recon.index[
                    np.isnan(_output_df_recon).argmin()
                ]
                df_temp = pd.DataFrame(
                    index=values.index,
                    columns=range(tstart_reconstructed - 1, first_valid_timestep + 1),
                    dtype=float,
                )
                df_temp.loc[:, tstart_reconstructed - 1] = values.loc[:, tend]
                df_temp.loc[:, first_valid_timestep] = values.loc[
                    :, sequence_timesteps[first_valid_timestep]
                ]
                df_temp = df_temp.interpolate(method="linear", axis=1)
                output_df.loc[
                    :, first_occurrences[tstart] : last_occurrences[tstart]
                ] = df_temp.loc[:, tstart_reconstructed:first_valid_timestep]

            output_df = output_df.apply(
                lambda row: np.interp(
                    sequence_timesteps.index,
                    row.index,
                    row.values,
                    left=np.nan,
                    right=np.nan,
                ),
                axis=1,
                result_type="expand",
            )
            if select_year_time_steps:
                sequence_timesteps = sequence_timesteps[
                    sequence_timesteps.isin(time_steps)
                ]

            if not output_df.empty:
                output_df = output_df[sequence_timesteps.index]
            else:
                output_df = values

        return self._rename_index(output_df.T.reset_index(drop=True).T.sort_index())

    def get_total(
        self,
        component_name: str,
        year: int | None = None,
        keep_raw: bool = False,
        index: Index | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Calculates the total values of a component for a specific scenario.

        :param component_name: Name of the component
        :param year: Filter the results by a given year
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        :param index: slicing index of the resulting dataframe
        :return: Total values of the component
        """
        series = self.get_values(
            component_name, index, keep_raw=keep_raw, rename_index=False
        )
        timestep_column, timestep_type = TimestepType.from_index_names(
            [str(name) for name in series.index.names]
        )

        if year is None:
            years = list(range(0, self.system.optimized_years))
        else:
            years = [self._convert_year2ts(year)]

        if timestep_type is None or type(series.index) is not pd.MultiIndex:
            if timestep_type is TimestepType.yearly:
                series = self._convert_ts2year(series)
            series = self._rename_index(series)
            return series

        if timestep_type is TimestepType.yearly:
            ans = series.unstack(timestep_column).sort_index()
            ans = ans.loc[:, years]
            ans = self._convert_ts2year(ans)
            ans = self._rename_index(ans)
            return ans

        timestep_duration = self._get_timestep_duration(timestep_type)

        unstacked_series = series.unstack(timestep_column).sort_index()
        total_value = unstacked_series.multiply(timestep_duration, axis=1)
        ans = pd.DataFrame(index=unstacked_series.index)

        for y in years:
            timesteps = self._get_time_steps(timestep_type, int(y))
            ans.insert(
                len(ans.columns),
                y,
                total_value[timesteps].sum(axis=1, skipna=False),
            )

        if "mf" in ans.index.names:
            ordered_levels = [str(i) for i in ans.index.names if i != "mf"] + ["mf"]
            ans = ans.reorder_levels(ordered_levels)
            ans = ans.sort_index(axis=0)

        ans = self._convert_ts2year(ans)
        ans = self._rename_index(ans)
        return ans

    def get_dual(
        self,
        component_name: str,
        year: int | None = None,
        discount_to_first_step: bool = True,
        keep_raw: bool = False,
        index: Index | None = None,
    ) -> pd.DataFrame | pd.Series | None:
        """Calculates the dual values of a component for a specific scenario.

        :param component_name: Name of the component.
        :param year: Filter the results by a given year.
        :param discount_to_first_step: Whether to discount the dual values
            to the first step.
        :param keep_raw: Keep the raw values of the rolling horizon optimization.
        :param index: Slicing index of the resulting dataframe.
        :return: Dual values of the component.
            Returns None if the duals were not saved for this scenario.
        """
        if not self.solver.save_duals:
            logger.warning("Duals were not saved for this scenario.")
            return None

        return self.get_full_ts(
            component_name,
            year=year,
            discount_to_first_step=discount_to_first_step,
            keep_raw=keep_raw,
            index=index,
        )

    def get_unit(
        self,
        component_name: str,
        convert_to_yearly_unit: bool,
    ) -> pd.Series | str | None:
        """Method that returns the unit of a component given its name and index.

        :param component_name: The name of the component.
        :param convert_to_yearly_unit: Whether to convert to yearly unit.
        :return: The unit of the component.
            Returns None if the component does not have a unit.
        """
        if component_name == "objective":
            if self.analysis.objective not in OBJECTIVE_FUNCTION_MAP:
                raise ValueError(
                    f"Invalid objective function {self.analysis.objective}"
                )
            component_name = OBJECTIVE_FUNCTION_MAP[self.analysis.objective]

        component_type = self.component_map.find_type(component_name)
        with pd.HDFStore(
            self.component_path / component_type.get_units_file_name(), mode="r"
        ) as store:
            if f"/{component_name}" not in store.keys():
                return None
            series = pd.read_hdf(store, component_name)
        assert isinstance(
            series, pd.Series
        ), f"Component {component_name} is not a series, but a {type(series)}."

        _, timestep_type = TimestepType.from_index_names(
            [str(name) for name in series.index.names]
        )
        unit_map = {
            unit: self._convert_to_pint_unit(
                unit, timestep_type, convert_to_yearly_unit
            )
            for unit in cast("pd.Series[str]", series.unique())
        }
        series = series.map(unit_map)
        series = series.sort_index()
        if series.size == 1 and series.index.name is None:
            return series.iloc[0]
        else:
            return self._rename_index(series)

    def get_doc(self, component_name: str) -> str | None:
        """Method that returns the docstring of a component given its name.

        :param component_name: The name of the component.
        :return: The docstring of the component.
        """
        component_type = self.component_map.find_type(component_name)
        doc_file = self.component_path / DOCS_FILENAME_MAP[component_type]
        with open(doc_file, "r") as f:
            docs = cast(dict[str, str], json.load(f))

        if component_name not in docs:
            return None

        doc = docs[component_name]
        if ";" in doc and ":" in doc:
            doc = "\n".join(v.replace(":", ": ") for v in doc.split(";"))
        return doc

    def get_index_names(self, component_name: str) -> list[str]:
        """Method that returns the index names of a component given its name."""
        component_type = self.component_map.find_type(component_name)
        file_path = self.component_path / component_type.get_file_name()
        store = NetCDF4DataStore.open(str(file_path), mode="r")
        try:
            return [str(dim) for dim in store.ds.variables[component_name].dimensions]
        finally:
            store.close()

    def _read_json_file(
        self,
        file_name: Path,
        obj_constr: type[T],
        model_constr: Callable[..., T] | None = None,
    ) -> T:
        """Reads a JSON file and returns an object of the specified type.

        :param file_name: The path to the JSON file.
        :param obj_constr: The constructor of the object to be created.
        :param model_constr: Optional constructor for the pydantic model.
            If provided, it will be used to create the object from the JSON data.
        :return: An object of the specified type.
        """
        if not file_name.exists():
            logger.warning(f"{file_name.name} does not exist for scenario {self.name}.")
            return obj_constr()

        with open(file_name, "r") as f:
            if model_constr is not None:
                return model_constr(**json.load(f))
            return obj_constr(**json.load(f))

    def _read_ureg(self) -> UnitRegistry:  # pyright:ignore[reportMissingTypeArgument]
        """Reads the unit definitions from the unit_definitions.txt file
        and returns a UnitRegistry from the pint library.

        :return: A pint.UnitRegistry object.
        """
        ureg: UnitRegistry = UnitRegistry(on_redefinition="ignore")
        unit_path = self.path / "unit_definitions.txt"
        if unit_path.exists():
            ureg.load_definitions(unit_path)  # pyright:ignore[reportUnusedCallResult]
        return ureg

    def _solution_version(self) -> int:
        """Load the version of the solution.
        The order in versions is important as the highest version should be checked
        last {v1,v2,...}.

        :param scenario: The scenario for which the version should be checked.

        :return: The version of the solution.
        """
        version_string = self.analysis.zen_garden_version
        if version_string is None:
            return 0
        for k, v in reversed(VERSION_MAP.items()):
            if self._compare_versions(v, version_string) <= 0:
                return k
        return 0

    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings.

        The comparison is done by checking version1 and version2.
        Each version is a string of *.*.* format, where the number of positions is
        arbitrary.

        :param version1: The first version.
        :param version2: The second version.

        :return: An integer that can be used in a comparison like this:
            if compare_versions(version1, version2) < 0:
                # version1 is less than version2
            elif compare_versions(version1, version2) > 0:
                # version1 is greater than version2
            else:
                # version1 is equal to version2
        """
        v1 = version1.replace("v", "").split(".")
        v2 = version2.replace("v", "").split(".")
        for i, j in zip(v1, v2, strict=False):
            if int(i) > int(j):
                return 1
            elif int(i) < int(j):
                return -1
        return 0

    def _convert_ts2year(self, df: FrameOrSeries) -> FrameOrSeries:
        """Converts the yearly ts column to the corresponding year."""
        df = df.copy()
        if isinstance(df, pd.Series):
            year_index = df.index
        else:
            year_index = df.columns
        assert pd.api.types.is_any_real_numeric_dtype(year_index), (
            f"DataFrame columns must be numeric to convert to year, not "
            f"{year_index.to_list()}."
        )
        ry = self.system.reference_year
        del_y = self.system.interval_between_years
        years = [ry + int(i) * del_y for i in year_index]
        if isinstance(df, pd.Series):
            df.index = years
        else:
            df.columns = years
        return df

    def _convert_year2ts(self, year: int) -> int:
        """Converts the year to the corresponding time step."""
        assert isinstance(year, int), f"Year must be an integer, not {type(year)}."
        ry = self.system.reference_year
        del_y = self.system.interval_between_years
        all_years = [ry + i * del_y for i in range(self.system.optimized_years)]
        if year in all_years:
            ts = (year - ry) // del_y
        elif year <= self.analysis.earliest_year_of_data and year in range(
            self.system.optimized_years
        ):
            warnings.warn(
                (
                    f"Selecting the yearly time steps ({year}) instead of the "
                    f"actual year ({ry + del_y * year}) is deprecated. Please use "
                    "the actual year."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            ts = year
        else:
            raise KeyError(f"Year {year} not in optimized years {all_years}.")
        return ts

    def _rename_index(self, df: FrameOrSeries) -> FrameOrSeries:
        """Renames the index of the dataframe."""
        if isinstance(df, pd.Series) and df.index.name == "scalar":
            return df.reset_index(drop=True)
        map = self.analysis.header_data_inputs
        df = df.copy()
        renamed_index = [
            map[str(idx)] if idx in map.keys() else idx for idx in df.index.names
        ]
        df.index.names = renamed_index
        return df

    def _combine_dataseries(
        self,
        series_dict: dict[int, pd.Series],
        timestep_column: str | None,
        timestep_type: TimestepType | None,
    ) -> pd.Series:
        """Method that combines the values when a solution is created without
        perfect foresight given a component, a scenario and a dictionary
        containing the name of the MF-data (Format: "MF_{year}").
        """
        series_to_concat: list[pd.Series] = []
        optimized_years = sorted(series_dict.keys())

        for i, year in enumerate(optimized_years):
            decision_horizon = tuple(
                range(year, optimized_years[i + 1])
                if i < len(optimized_years) - 1
                else [year]
            )

            current_mf = series_dict[year]
            if current_mf.empty:
                continue

            if timestep_type is None:
                series_to_concat.append(current_mf)
                break
            elif timestep_type is TimestepType.yearly:
                year_series = current_mf[
                    current_mf.index.get_level_values("set_years").isin(
                        decision_horizon
                    )
                ]
                series_to_concat.append(year_series)
            else:
                assert timestep_column is not None
                time_steps = self._get_timesteps_of_years(
                    timestep_type, decision_horizon
                )
                year_series = current_mf[
                    current_mf.index.get_level_values(timestep_column).isin(time_steps)
                ]
                series_to_concat.append(year_series)

        if len(series_to_concat) == 0:
            return pd.Series(dtype=float)
        return pd.concat(series_to_concat)

    def _concatenate_raw_series(self, series_dict: dict[int, pd.Series]) -> pd.Series:
        """Concatenate the raw values when a solution is created
        without perfect foresight given a component, a scenario and a
        dictionary containing the name of the MF-data (Format: "MF_{year}").
        The raw values are not combined, i.e., the data is kept for all the
        foresight steps.
        """
        if not series_dict:
            raise ValueError("series_dict must not be empty")

        index_names = next(iter(series_dict.values())).index.names
        series = pd.concat(series_dict, names=["mf", *index_names])
        return series.sort_index(level="mf")

    def _convert_to_pint_unit(
        self,
        u: str,
        timestep_type: TimestepType | None,
        convert_to_yearly_unit: bool,
    ) -> str:
        """Converts a string to a pint unit.

        :param u: The unit string.
        :param component_name: The name of the component.
        :param convert_to_yearly_unit: Whether to convert to yearly unit.
        :return: The pint unit string.
        """
        try:
            quantity = self.ureg.parse_expression(u)
            if convert_to_yearly_unit and timestep_type is TimestepType.operational:
                quantity = quantity * self.ureg.h
            return f"{quantity.units:~D}"

        # if the unit is not in the pint registry, change the string manually
        # (normally when the unit_definition.txt is not saved)
        # TODO: fix cases like conversion_factor,
        #       which returns an object and not a string
        except Exception:
            if convert_to_yearly_unit and timestep_type is TimestepType.operational:
                if u.endswith(" / hour"):
                    return u.replace(" / hour", "")
                return f"{u} * hour"
            return u

    def _annuity(self, discount_to_first_step: bool = True) -> pd.Series:
        """Discounts the duals.

        Args:
            discount_to_first_step: apply annuity to first year of interval or
                entire interval

        Returns:
            annuity of the duals
        """
        system = self.system
        discount_rate = cast(float, self.get_values("discount_rate").squeeze())

        years = list(range(0, self.system.optimized_years))
        annuity = pd.Series(index=years, dtype=float)
        optimized_years = self._get_optimized_years()

        for year in years:
            # closest year in optimized years that is smaller than year
            start_year = [y for y in optimized_years if y <= year][-1]
            interval_between_years = system.interval_between_years
            interval_between_years_this_year = (
                self.system.interval_between_years if year != years[-1] else 1
            )

            if discount_to_first_step:
                annuity[year] = interval_between_years_this_year * (
                    (1 / (1 + discount_rate))
                    ** (interval_between_years * (year - start_year))
                )
            else:
                annuity[year] = sum(
                    (
                        (1 / (1 + discount_rate))
                        ** (
                            interval_between_years * (year - start_year)
                            + _intermediate_time_step
                        )
                    )
                    for _intermediate_time_step in range(
                        0, interval_between_years_this_year
                    )
                )
        return annuity

    def _get_timesteps_of_years(
        self, timestep_type: TimestepType, years: tuple[int, ...]
    ) -> set[int]:
        """Method that returns the timesteps of the scenario for a given year.

        :param timestep_type: The type of the timestep.
        :param years: The years for which the timesteps should be returned.
        :return: A list of timesteps.
        """
        assert timestep_type is not TimestepType.yearly

        if timestep_type is TimestepType.storage:
            time_step_yearly = self.time_steps.time_steps_year2storage
        elif timestep_type is TimestepType.operational:
            time_step_yearly = self.time_steps.time_steps_year2operation
        return set(
            itertools.chain.from_iterable(time_step_yearly[str(year)] for year in years)
        )

    def _get_sequence_time_steps(self, timestep_type: TimestepType) -> pd.Series:
        """Method that returns the sequence time steps of a scenario.

        Args:
            scenario
            timestep_type
        """
        if timestep_type is TimestepType.operational:
            ans = self.time_steps.operation
        elif timestep_type is TimestepType.storage:
            ans = self.time_steps.storage
        elif timestep_type is TimestepType.yearly:
            ans = self.time_steps.yearly
        return pd.Series(ans)

    def _get_optimized_years(self) -> list[int]:
        """Method that returns the years for which the solution was optimized.
        Raises an exception if it's the solution is in an old format.
        """
        return self.time_steps.optimized_time_steps

    def _get_timestep_duration(self, timestep_type: TimestepType) -> pd.Series:
        """The timestep duration is stored as any other component, the only thing
        is to define the correct name depending on the component timestep type.
        """
        if timestep_type is TimestepType.operational:
            raw_data = self.time_steps.time_steps_operation_duration
        else:
            raw_data = self.time_steps.time_steps_storage_duration
        ans = pd.Series(raw_data)
        ans.index = ans.index.astype(int)
        return ans.astype(int)

    def _get_time_steps_storage_level_startend_year(self) -> dict[int, int]:
        """Return time steps that define the start and end of the storage level.

        :param scenario: scenario name.
        """
        ans = self.time_steps.time_steps_storage_level_startend_year
        return {int(k): int(v) for k, v in ans.items()}

    def _get_time_steps(self, timestep_type: TimestepType, year: int) -> pd.Series:
        """THe timesteps are stored in a file HDF-File called
        dict_all_sequence_time_steps saved for each scenario. The name of the
        dataframe depends on the timestep type.
        """
        if timestep_type is TimestepType.operational:
            ans = self.time_steps.time_steps_year2operation
        else:
            ans = self.time_steps.time_steps_year2storage
        return pd.Series(ans[str(year)])
