"""Scenario reader for output-version-1 ZEN-garden result folders.

Output version 1 covers result folders written by ZEN-garden ``>= 2.0.14`` and
``< 2.2.15``. The component/unit payload is stored exactly as in output
version 2 (a ``value``/``units`` DataFrame, or a bare Series when there is no
unit), so :class:`ScenarioV1` subclasses
:class:`~zen_garden.postprocess.results.scenario_v2.ScenarioV2`.

The only additional difference is the time-step file: version 1 folders written
before ZEN-garden 2.2.9 do not persist ``time_steps_storage_level_startend_year``.
Matching the legacy solution loader, that mapping is reconstructed here from the
storage sequence and the number of (unaggregated) time steps per year, and
:meth:`time_steps` tolerates the key being absent from the JSON.
"""

import json
from dataclasses import fields

from zen_garden.postprocess.results.scenario_v2 import ScenarioV2
from zen_garden.postprocess.results.timestep_map import TimestepMap
from zen_garden.postprocess.results.timestep_type import TimestepType


class ScenarioV1(ScenarioV2):
    """A :class:`ScenarioV2` that also reconstructs the storage start/end map."""

    #: Output version implemented by this reader.
    OUTPUT_VERSION = 1

    @property
    def time_steps(self) -> TimestepMap:
        """Load the time-step map, tolerating the pre-2.2.9 layout.

        Older output-version-1 folders do not store
        ``time_steps_storage_level_startend_year``; it is filled with an empty
        mapping here and recomputed in
        :meth:`_get_time_steps_storage_level_startend_year`. Unknown keys are
        dropped so that additional legacy entries do not break construction.
        """
        if self._time_steps is None:
            matches = list(self.path.glob("dict_all_sequence_time_steps*.json"))
            assert (
                len(matches) == 1
            ), f"Expected exactly one time steps file, found {len(matches)}"
            with open(matches[0], "r") as f:
                raw = json.load(f)
            known = {field.name for field in fields(TimestepMap)}
            data = {key: raw[key] for key in known if key in raw}
            data.setdefault("time_steps_storage_level_startend_year", {})
            self._time_steps = TimestepMap(**data)
        return self._time_steps

    def _get_time_steps_storage_level_startend_year(self) -> dict[int, int]:
        """Reconstruct the storage-level start/end time steps for each year.

        Output version 1 does not persist this mapping; it is derived from the
        storage sequence and the (unaggregated) number of time steps per year,
        matching the legacy solution loader.
        """
        sequence = self._get_sequence_time_steps(TimestepType.storage)
        steps_per_year = self.system.unaggregated_time_steps_per_year
        startend: dict[int, int] = {}
        for year in range(self.system.optimized_years):
            start_idx = year * steps_per_year
            end_idx = (year + 1) * steps_per_year - 1
            startend[int(sequence.iloc[start_idx])] = int(sequence.iloc[end_idx])
        return startend
