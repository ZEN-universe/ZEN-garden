"""Scenario reader for output-version-2 ZEN-garden result folders.

Output version 2 covers result folders written by ZEN-garden ``>= 2.2.15`` and
``< 2.9.2``. The folder layout is the same as output version 3 (per-type HDF5
``*_dict.h5`` files, one component per key, metadata in the HDF5 group
attributes, JSON time-step and config files), with a single difference in how a
component and its unit are stored:

* **v2** - a component with a unit is a ``pandas.DataFrame`` under its key with a
  ``value`` column and a ``units`` column; a component without a unit is a bare
  ``pandas.Series``.
* **v3** - a component is always a bare ``pandas.Series`` and its unit series (if
  any) lives under a separate ``<key>_units`` key.

"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zen_garden.postprocess.results.scenario_v3 import V3_FILE_NAME_MAP, ScenarioV3

#: Name of the value column inside a v1/v2 component DataFrame.
_VALUE_COLUMN = "value"
#: Name of the unit column inside a v1/v2 component DataFrame.
_UNITS_COLUMN = "units"


class ScenarioV2(ScenarioV3):
    """A :class:`ScenarioV3` that reads the output-version-2 payload layout."""

    #: Output version implemented by this reader.
    OUTPUT_VERSION = 2

    def _load_unit_series(self, component_name: str) -> pd.Series | None:
        """Read the unit series from the ``units`` column of the component frame.

        Output versions 1 and 2 keep the unit strings as a second column of the
        component's DataFrame rather than in a separate ``<name>_units`` key.

        :param component_name: The name of the component.
        :return: The unit series, or ``None`` if the component has no unit.
        """
        component_type = self.component_map.find_type(component_name)
        file_path = self.component_path / V3_FILE_NAME_MAP[component_type]
        frame = pd.read_hdf(file_path, component_name)
        if not isinstance(frame, pd.DataFrame) or _UNITS_COLUMN not in frame.columns:
            return None
        return frame[_UNITS_COLUMN]

    @staticmethod
    def _read_component_series(file_path: Path, component_name: str) -> pd.Series:
        """Read a component's values as a ``pandas.Series``.

        Output versions 1 and 2 store a component with units as a DataFrame with
        a ``value`` column (plus a ``units`` column); a component without units
        is stored as a bare Series. Version 3 switched to always storing a bare
        Series.

        :param file_path: Path to the ``*_dict.h5`` file.
        :param component_name: The HDF5 key of the component.
        :return: The component values as a series.
        """
        raw = pd.read_hdf(file_path, component_name)
        if isinstance(raw, pd.DataFrame):
            if _VALUE_COLUMN in raw.columns:
                value: Any = raw[_VALUE_COLUMN]
            else:
                data_columns = [c for c in raw.columns if c != _UNITS_COLUMN]
                value = (
                    raw[data_columns[0]] if len(data_columns) == 1 else raw.squeeze()
                )
        else:
            value = raw
        if isinstance(value, (np.floating, np.integer, float, int, str)):
            value = pd.Series([value], index=getattr(raw, "index", None))
        if not isinstance(value, pd.Series):
            raise TypeError(
                f"Component {component_name} in {file_path.name} could not be "
                f"read as a pandas Series (got {type(value)})."
            )
        return value
