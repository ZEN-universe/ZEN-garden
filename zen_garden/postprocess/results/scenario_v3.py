"""Scenario reader for pre-netCDF ZEN-garden result folders (output version 3).

Result folders written by ZEN-garden ``< 3.0.0`` (output version 3 - the last
on-disk format supported by ``solution_loader.py`` on the ``main`` branch before
the switch to netCDF) store every component in per-type HDF5 files
(``var_dict.h5``, ``param_dict.h5``, ``set_dict.h5``, ``dual_dict.h5`` and,
optionally, ``reduced_costs_dict.h5``) rather than the ``*.nc`` files plus
``component_map.json`` / ``*_docs.json`` used by the current
:class:`~zen_garden.postprocess.results.scenario.Scenario`.

Within one of those HDF5 files each component is a ``pandas.Series`` stored under
its own key, its unit series (if any) is stored under ``<key>_units``, and its
metadata lives in the HDF5 attributes of the component group (``docstring``,
``index_names``, ``has_units``, ``name``).

:class:`ScenarioV3` inherits from :class:`Scenario` and overrides only the
members that read the result files directly: :attr:`component_map` (there is no
``component_map.json``), :meth:`get_raw_values`, :meth:`get_unit` (via
:meth:`_load_unit_series`), :meth:`get_doc` and :meth:`get_index_names`. The
config JSONs, ``unit_definitions.txt`` and ``dict_all_sequence_time_steps*.json``
are byte-for-byte compatible and their accessors are inherited unchanged.

Legacy folders also name the index levels with the *friendly* names (``year``,
``node``, ``time_operation``, ...) instead of the internal ``set_*`` names the
current :class:`Scenario` machinery expects, so :meth:`get_raw_values` and
:meth:`get_unit` normalise them via :meth:`_to_internal_index_names` (using the
inverse of ``analysis.header_data_inputs``) before handing the series to the
inherited logic, which renames them back on the way out. Without this,
``get_total`` / ``get_full_ts`` would neither unstack the time dimension nor
convert the time-step index to calendar years.

The raw file access is isolated in a few small helpers
(:meth:`_read_component_series`, :meth:`_load_unit_series`,
:meth:`_to_internal_index_names`, :meth:`time_steps`) so that the readers for
the even older output versions 1 and 2
(:mod:`~zen_garden.postprocess.results.scenario_v2` /
:mod:`~zen_garden.postprocess.results.scenario_v1`) can subclass
:class:`ScenarioV3` and override just those.
"""

import logging
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd

from zen_garden.config import HeaderDataInputs
from zen_garden.postprocess.results.component_map import ComponentMap
from zen_garden.postprocess.results.component_type import ComponentType
from zen_garden.postprocess.results.scenario import (
    OBJECTIVE_FUNCTION_MAP,
    Index,
    Scenario,
    _build_query,
)
from zen_garden.postprocess.results.timestep_type import TimestepType

logger = logging.getLogger(__name__)

#: Maps a component type to the HDF5 file that holds it in an output-version-3
#: result folder.
V3_FILE_NAME_MAP: dict[ComponentType, str] = {
    ComponentType.sets: "set_dict.h5",
    ComponentType.variable: "var_dict.h5",
    ComponentType.parameter: "param_dict.h5",
    ComponentType.dual: "dual_dict.h5",
    ComponentType.reduced_costs: "reduced_costs_dict.h5",
}

#: Suffix used for the per-component unit series stored alongside the data.
_UNITS_SUFFIX = "_units"


def _decode_hdf_attr(value: Any) -> str:
    """Decode an HDF5 attribute that may be ``bytes``, ``str`` or a 0-d array.

    :param value: The raw attribute value returned by ``h5py``.
    :return: The attribute as a plain string.
    """
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class ScenarioV3(Scenario):
    """A :class:`Scenario` that reads output-version-3 (pre-netCDF) folders."""

    #: Output version implemented by this reader.
    OUTPUT_VERSION = 3

    @property
    def output_version(self) -> int:
        """Return the output version implemented by this reader."""
        return self.OUTPUT_VERSION

    @property
    def component_map(self) -> ComponentMap:
        """Return the component map, rebuilt from the ``*_dict.h5`` keys.

        The pre-netCDF format has no ``component_map.json``; a component's type
        is given by the file it lives in. ``<component>_units`` keys are unit
        side-cars rather than components and are filtered out.
        """
        if self._component_map is None:
            names_by_type: dict[ComponentType, list[str]] = {}
            for component_type, file_name in V3_FILE_NAME_MAP.items():
                file_path = self.component_path / file_name
                if not file_path.exists():
                    names_by_type[component_type] = []
                    continue
                with h5py.File(file_path, "r") as h5_file:
                    names_by_type[component_type] = [
                        key for key in h5_file.keys() if not key.endswith(_UNITS_SUFFIX)
                    ]
            self._component_map = ComponentMap(
                sets=names_by_type.get(ComponentType.sets, []),
                variable=names_by_type.get(ComponentType.variable, []),
                parameter=names_by_type.get(ComponentType.parameter, []),
                dual=names_by_type.get(ComponentType.dual, []),
                reduced_cost=names_by_type.get(ComponentType.reduced_costs, []),
            )
        return self._component_map

    def get_raw_values(
        self,
        component_name: str,
        index: Index | None = None,
        rename_index: bool = True,
        mf_folder: str | None = None,
    ) -> pd.Series:
        """Get the values for a given component filtered by the index.

        :param component_name: The name of the component.
        :param index: The index used to slice the series unless the component
            is a set.
        :param rename_index: Whether to rename the index of the series.
        :param mf_folder: Optional rolling-horizon subfolder (``MF_<n>``).
        :return: The pandas series object.
        """
        component_type = self.component_map.find_type(component_name)
        sub_folder: Path = (
            Path(".")
            if mf_folder is None or not self.has_rh
            else Path("..") / mf_folder
        )
        file_path = self.component_path / sub_folder / V3_FILE_NAME_MAP[component_type]

        series = self._read_component_series(file_path, component_name)
        series = self._to_internal_index_names(series)
        if component_type is ComponentType.sets:
            return series

        header_data_inputs = self.analysis.header_data_inputs
        header_map = (
            header_data_inputs
            if isinstance(header_data_inputs, dict)
            else header_data_inputs.model_dump()
        )
        raw_dims = tuple(str(dim) for dim in series.index.names)
        query = _build_query(raw_dims, header_map, index)
        ans = self._apply_query(series, query, raw_dims).dropna()
        if rename_index:
            ans = self._rename_index(ans)
        return ans

    def get_unit(
        self,
        component_name: str,
        convert_to_yearly_unit: bool,
    ) -> pd.Series | str | None:
        """Return the unit of a component.

        :param component_name: The name of the component.
        :param convert_to_yearly_unit: Whether to convert to a yearly unit.
        :return: The unit of the component.
            Returns ``None`` if the component does not have a unit.
        """
        if component_name == "objective":
            if self.analysis.objective not in OBJECTIVE_FUNCTION_MAP:
                raise ValueError(
                    f"Invalid objective function {self.analysis.objective}"
                )
            component_name = OBJECTIVE_FUNCTION_MAP[self.analysis.objective]

        series = self._load_unit_series(component_name)
        if series is None:
            return None
        series = self._to_internal_index_names(series)

        # Post-processing below is identical to Scenario.get_unit; only the
        # location of the raw unit series (see ``_load_unit_series``) differs
        # between the formats.
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
        return self._rename_index(series)

    def get_doc(self, component_name: str) -> str | None:
        """Return the docstring of a component, read from the HDF5 attribute.

        :param component_name: The name of the component.
        :return: The docstring, or ``None`` if the component has none.
        """
        component_type = self.component_map.find_type(component_name)
        file_path = self.component_path / V3_FILE_NAME_MAP[component_type]
        with h5py.File(file_path, "r") as h5_file:
            if component_name not in h5_file:
                return None
            raw_doc = h5_file[component_name].attrs.get("docstring")

        if raw_doc is None:
            return None
        doc = _decode_hdf_attr(raw_doc)
        if ";" in doc and ":" in doc:
            doc = "\n".join(v.replace(":", ": ") for v in doc.split(";"))
        return doc

    def get_index_names(self, component_name: str) -> list[str]:
        """Return the index names of a component, read from the HDF5 attribute.

        :param component_name: The name of the component.
        :return: The list of index (dimension) names.
        """
        component_type = self.component_map.find_type(component_name)
        file_path = self.component_path / V3_FILE_NAME_MAP[component_type]
        with h5py.File(file_path, "r") as h5_file:
            raw_names = h5_file[component_name].attrs["index_names"]
        index_names = _decode_hdf_attr(raw_names)
        return index_names.split(",") if index_names else []

    def _load_unit_series(self, component_name: str) -> pd.Series | None:
        """Read the raw unit series for a component from its ``<name>_units`` key.

        :param component_name: The name of the component.
        :return: The unit series, or ``None`` if the component has no unit.
        """
        component_type = self.component_map.find_type(component_name)
        file_path = self.component_path / V3_FILE_NAME_MAP[component_type]
        units_key = f"{component_name}{_UNITS_SUFFIX}"

        with pd.HDFStore(file_path, mode="r") as store:
            if f"/{units_key}" not in store.keys():
                return None
            series = pd.read_hdf(store, units_key)

        assert isinstance(series, pd.Series), (
            f"Component {component_name} units are not a series, "
            f"but a {type(series)}."
        )
        return series

    def _friendly_to_internal_index(self) -> dict[str, str]:
        """Map on-disk (friendly) index names to internal ``set_*`` names.

        Built by inverting ``analysis.header_data_inputs`` (internal -> friendly).
        The folder's own mapping takes precedence so that the names produced here
        round-trip cleanly through :meth:`_rename_index`; the current
        :class:`HeaderDataInputs` schema fills in any friendly name the folder
        did not record (``header_data_inputs`` is loaded without validation and
        may be partial). When several dimensions share a friendly name the first
        wins - the choice is irrelevant because :meth:`_rename_index` collapses
        them back to the same friendly name.

        The yearly dimension is the one case where an old folder's internal name
        (``set_time_steps_yearly``) differs from today's (``set_years``); both are
        recognised by :meth:`TimestepType.from_index_names`, so either resolves to
        :attr:`TimestepType.yearly` downstream.

        :return: A mapping ``friendly name -> internal name``.
        """
        header_data_inputs = self.analysis.header_data_inputs
        folder_map = (
            header_data_inputs
            if isinstance(header_data_inputs, dict)
            else header_data_inputs.model_dump()
        )
        inverse: dict[str, str] = {}
        for internal, friendly in folder_map.items():
            inverse.setdefault(str(friendly), str(internal))
        for internal, friendly in HeaderDataInputs().model_dump().items():
            inverse.setdefault(str(friendly), str(internal))
        return inverse

    def _to_internal_index_names(self, series: pd.Series) -> pd.Series:
        """Rename a series' index levels from friendly to internal names.

        Legacy result folders store friendly index names (``year``, ``node``,
        ...); the current :class:`Scenario` machinery (``get_total``,
        ``get_full_ts``, ``TimestepType.from_index_names``, ...) expects the
        internal ``set_*`` names. Levels that are already internal (or unknown)
        are left untouched, and :meth:`_rename_index` maps everything back to the
        friendly names on the way out.

        :param series: The series whose index levels should be normalised.
        :return: The series with normalised index level names.
        """
        mapping = self._friendly_to_internal_index()
        new_names = [mapping.get(str(name), name) for name in series.index.names]
        if new_names != list(series.index.names):
            series = series.copy()
            series.index = series.index.set_names(new_names)
        return series

    @staticmethod
    def _read_component_series(file_path: Path, component_name: str) -> pd.Series:
        """Read a single component as a ``pandas.Series`` from an HDF5 file.

        :param file_path: Path to the ``*_dict.h5`` file.
        :param component_name: The HDF5 key of the component.
        :return: The component values as a series.
        """
        raw = pd.read_hdf(file_path, component_name)
        value: Any = raw.squeeze() if isinstance(raw, pd.DataFrame) else raw
        if isinstance(value, (np.floating, np.integer, float, int, str)):
            value = pd.Series([value], index=getattr(raw, "index", None))
        if not isinstance(value, pd.Series):
            raise TypeError(
                f"Component {component_name} in {file_path.name} could not be "
                f"read as a pandas Series (got {type(value)})."
            )
        return value

    @staticmethod
    def _apply_query(
        series: pd.Series,
        query: dict[str, str],
        raw_dims: tuple[str, ...],
    ) -> pd.Series:
        """Filter a series with the query produced by ``_build_query``.

        :class:`Scenario` applies that query to an ``xarray.DataArray``; the
        pre-netCDF format is read straight into a ``pandas.Series``, so the same
        boolean expressions are evaluated with ``DataFrame.query`` instead.

        :param series: The unfiltered component series.
        :param query: Mapping of raw dimension name to a boolean expression.
        :param raw_dims: The series index level names, in order.
        :return: The filtered series.
        """
        if not query:
            return series
        frame = series.reset_index(name="__value__")
        expr = " & ".join(f"({term})" for term in query.values())
        frame = frame.query(expr, engine="python")
        result = frame.set_index(list(raw_dims))["__value__"]
        result.name = series.name
        return result
