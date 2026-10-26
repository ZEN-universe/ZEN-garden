"""This file implements a helper class to deal with timesteps."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeStepsDicts(object):
    """Helper class to deal with time steps of the optimization setup.

    This class implements some simple helper functions that can deal with the time
    steps of the optimization setup. It is very similar to EnergySystem functions and
    is meant to avoid the import of packages that can cause conflicts.
    """

    time_steps_operation: list | None
    time_steps_storage: list | None
    sequence_time_steps_operation: np.ndarray | None
    sequence_time_steps_storage: np.ndarray | None
    sequence_time_steps_yearly: np.ndarray | None
    time_steps_operation_duration: dict | None
    time_steps_storage_duration: dict | None
    time_steps_operation2year: dict | None
    time_steps_year2operation: dict | None
    time_steps_storage2year: dict | None
    time_steps_year2storage: dict | None
    time_steps_energy2power: dict | None
    time_steps_storage_level_startend_year: dict | None
    representative_hours: list[int] | None
    representative_hours_storage: dict[int, list[int]] | None
    weight_representative_hours: pd.Series | None
    duration_representative_hours_storage: pd.Series | None
    operation_step_to_pair: dict[int, tuple[int, int]] | None
    storage_step_to_pair: dict[int, tuple[int, int]] | None
    operation_pair_to_step: dict[tuple[int, int], int] | None
    storage_pair_to_step: dict[tuple[int, int], int] | None
    storage_to_operation: pd.Series | None
    previous_storage_time_step: pd.Series | None
    storage_coupling_mask: pd.Series | None

    def __init__(self, dict_all_sequence_time_steps=None):
        """Sets all dicts of sequences of time steps.

        :param dict_all_sequence_time_steps: dict of all dict_sequence_time_steps
        """
        if dict_all_sequence_time_steps is None:
            self.time_steps_operation = None
            self.time_steps_storage = None
            self.sequence_time_steps_operation = None
            self.sequence_time_steps_storage = None
            self.sequence_time_steps_yearly = None
            self.time_steps_operation_duration = None
            self.time_steps_storage_duration = None

            self.time_steps_operation2year = None
            self.time_steps_year2operation = None
            self.time_steps_storage2year = None
            self.time_steps_year2storage = None

            self.time_steps_energy2power = None

            self.time_steps_storage_level_startend_year = None

            self.representative_hours = None
            self.representative_hours_storage = None
            self.weight_representative_hours = None
            self.duration_representative_hours_storage = None
            self.operation_step_to_pair = None
            self.storage_step_to_pair = None
            self.operation_pair_to_step = None
            self.storage_pair_to_step = None
            self.storage_to_operation = None
            self.previous_storage_time_step = None
            self.storage_coupling_mask = None

        # set the params if provided
        else:
            el_op = [
                el
                for el in dict_all_sequence_time_steps["operation"].keys()
                if "storage_level" not in el
            ][0]
            el_stor = [
                el
                for el in dict_all_sequence_time_steps["operation"].keys()
                if "storage_level" in el
            ][0]
            self.sequence_time_steps_operation = dict_all_sequence_time_steps[
                "operation"
            ][el_op]
            self.sequence_time_steps_storage = dict_all_sequence_time_steps[
                "operation"
            ][el_stor]
            self.sequence_time_steps_yearly = dict_all_sequence_time_steps["yearly"][
                None
            ]
            self.set_time_steps_operation2year_both_dir()
            self.set_time_steps_storage2year_both_dir()

    def set_aggregated_time_steps(self, tsa):
        """Sets the aggregated time steps types.

        :param tsa: time series aggregation object
        """
        self.time_steps_operation = list(tsa.set_hours)
        self.time_steps_operation_duration = tsa.time_steps_duration
        self.sequence_time_steps_operation = tsa.sequence_time_steps

    def get_sequence_time_steps(self, time_step_type: str = "operation"):
        """Get sequence ot time steps of element.

        :param time_step_type: type of time step (operation, storage or yearly)
        :return: sequence_time_steps: list of time steps corresponding to base time step
        """
        if time_step_type == "operation":
            return self.sequence_time_steps_operation
        elif time_step_type == "storage":
            return self.sequence_time_steps_storage
        elif time_step_type == "yearly":
            return self.sequence_time_steps_yearly
        else:
            raise KeyError(f"Time step type {time_step_type} is incorrect")

    def get_sequence_time_steps_dict(self):
        """Returns all dicts of sequence of time steps.

        :return: dict_all_sequence_time_steps: dict of all dict_sequence_time_steps
        """
        dict_all_sequence_time_steps = {
            "operation": self.sequence_time_steps_operation,
            "storage": self.sequence_time_steps_storage,
            "yearly": self.sequence_time_steps_yearly,
        }
        return dict_all_sequence_time_steps

    def encode_time_step(
        self, base_time_steps: int | np.ndarray, time_step_type: str = "operation"
    ):
        """Encodes baseTimeStep to timeStep of element.

        Encodes baseTimeStep, i.e., retrieves the time step of an element corresponding
        to baseTimeStep of model. baseTimeStep of model --> timeStep of element.

        :param base_time_steps: base time step of model for which the corresponding
            time index is extracted
        :param time_step_type: invest or operation. Only relevant for technologies
        :return: outputTimeStep: time step of element
        """
        sequence_time_steps = self.get_sequence_time_steps(time_step_type)
        # get time step duration
        if np.all(base_time_steps >= 0):
            assert sequence_time_steps is not None
            element_time_step = np.unique(sequence_time_steps[base_time_steps])
        else:
            element_time_step = np.array([-1])
        return element_time_step

    def decode_time_step(
        self, element_time_step: int, time_step_type: str = "operation"
    ):
        """Decodes timeStep to baseTimeStep of model.

        Decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the
        variableTimeStep of a element. timeStep of element --> baseTimeStep of model.

        :param element_time_step: time step of element
        :param time_step_type: invest or operation. Only relevant for technologies,
            None for carrier
        :return: baseTimeStep of model
        """
        sequence_time_steps = self.get_sequence_time_steps(time_step_type)
        # find where element_time_step in sequence of element time steps
        base_time_steps = np.argwhere(sequence_time_steps == element_time_step)
        return base_time_steps

    def calculate_time_step_duration(self, input_time_steps, base_time_steps):
        """Calculates (equidistant) time step durations for input time steps.

        :param input_time_steps: input time steps
        :param base_time_steps: manual list of base time steps
        :return: time_step_duration_dict: dict with duration of each time step
        """
        duration_input_time_steps = len(base_time_steps) / len(input_time_steps)
        time_step_duration_dict = {
            time_step: int(duration_input_time_steps) for time_step in input_time_steps
        }
        if not duration_input_time_steps.is_integer():
            logger.warning(
                f"The duration of each time step {duration_input_time_steps} of input "
                f"time steps {input_time_steps} does not evaluate to an integer. \n"
                f"The duration of the last time step is set to compensate for "
                f"the difference"
            )
            duration_last_time_step = len(base_time_steps) - sum(
                time_step_duration_dict[key]
                for key in time_step_duration_dict
                if key != input_time_steps[-1]
            )
            time_step_duration_dict[input_time_steps[-1]] = duration_last_time_step
        return time_step_duration_dict

    def set_time_steps_operation2year_both_dir(self):
        """Converts operational time steps to investment/yearly time steps."""
        sequence_operation = self.sequence_time_steps_operation
        sequence_yearly = self.sequence_time_steps_yearly
        time_steps_combi_operation = np.vstack(
            pd.unique(pd.Series(zip(sequence_operation, sequence_yearly, strict=False)))
        ).T
        time_steps_combi_operation = np.sort(time_steps_combi_operation, axis=1)
        # calculate operation2year
        time_steps_operation2year = {
            key: val
            for key, val in zip(
                time_steps_combi_operation[0, :],
                time_steps_combi_operation[1, :],
                strict=False,
            )
        }
        self.time_steps_operation2year = time_steps_operation2year
        # calculate year2operation
        time_steps_year2operation = {}
        for year in pd.unique(time_steps_combi_operation[1]):
            time_steps_year2operation[year] = time_steps_combi_operation[
                0, time_steps_combi_operation[1] == year
            ]
        self.time_steps_year2operation = time_steps_year2operation

    def set_time_steps_storage2year_both_dir(self):
        """Converts storage time steps to investment/yearly time steps."""
        sequence_storage = self.sequence_time_steps_storage
        sequence_yearly = self.sequence_time_steps_yearly
        time_steps_combi_storage = np.vstack(
            pd.unique(pd.Series(zip(sequence_storage, sequence_yearly, strict=False)))
        ).T
        # calculate storage2year
        time_steps_storage2year = {
            key: val
            for key, val in zip(
                time_steps_combi_storage[0, :],
                time_steps_combi_storage[1, :],
                strict=False,
            )
        }
        self.time_steps_storage2year = time_steps_storage2year
        # calculate year2storage
        time_steps_year2storage = {}
        for year in pd.unique(time_steps_combi_storage[1]):
            time_steps_year2storage[year] = time_steps_combi_storage[
                0, time_steps_combi_storage[1] == year
            ]
        self.time_steps_year2storage = time_steps_year2storage

    def set_time_steps_storage_startend(self, system):
        """Matches last time step of the year to the first for storage level time steps.

        sets the dict of matching the last time step of the year in the
        storage level domain to the first.

        :param system: dictionary defining the system
        """
        assert self.sequence_time_steps_storage is not None
        unaggregated_time_steps = system.unaggregated_time_steps_per_year
        sequence_time_steps = self.sequence_time_steps_storage
        counter = 0
        time_steps_start = []
        time_steps_end = []
        assert (
            system.interval_between_years == 1 or not system.multiyear_periodicity
        ), "The interval between years should be 1 for multiyear storage periodicity."
        if not system.multiyear_periodicity:
            while counter < len(sequence_time_steps):
                time_steps_start.append(sequence_time_steps[counter])
                counter += unaggregated_time_steps
                time_steps_end.append(sequence_time_steps[counter - 1])
            self.time_steps_storage_level_startend_year = {
                start: end
                for start, end in zip(time_steps_start, time_steps_end, strict=False)
            }
        else:
            self.time_steps_storage_level_startend_year = {
                self.sequence_time_steps_storage[0]: self.sequence_time_steps_storage[
                    -1
                ]
            }

    def get_time_steps_year2operation(self, year=None):
        """Gets conversion dict from invest time steps to operation time steps of techs.

        :param year: year of interest
        :return: time_steps_year2operation of the specified element (at specified year)
        """
        assert (
            self.time_steps_year2operation is not None
        ), "The time_steps_year2operation dict is not set."
        if year is None:
            return self.time_steps_year2operation
        else:
            return self.time_steps_year2operation[year]

    def get_time_steps_storage2year(self):
        """Gets conversion dict from storage time steps to invest time steps of techs.

        :return: time_steps_storage2year of the specified element
        """
        return self.time_steps_storage2year

    def get_time_steps_year2storage(self, year=None):
        """Gets conversion dict from invest time steps to storage time steps of techs.

        :param year: year of interest
        :return: time_steps_year2storage of the specified element (at specified year)
        """
        assert (
            self.time_steps_year2storage is not None
        ), "The time_steps_year2storage dict is not set."
        if year is None:
            return self.time_steps_year2storage
        else:
            return self.time_steps_year2storage[year]

    def get_time_steps_storage_startend(self, time_step):
        """Gets conversion dict from operation time steps to invest time steps of techs.

        :param time_step: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        assert self.time_steps_storage_level_startend_year is not None
        if time_step in self.time_steps_storage_level_startend_year.keys():
            return self.time_steps_storage_level_startend_year[time_step]
        else:
            return None

    def get_previous_storage_time_step(self, time_step):
        """Gets the storage time step before in the sequence.

        :param time_step: current time step
        :return: previous_time_step: previous time step
        """
        assert self.sequence_time_steps_storage is not None
        sequence = self.sequence_time_steps_storage
        previous_time_step = sequence[np.where(sequence == time_step)[0] - 1][0]
        return previous_time_step

    def decode_yearly_time_steps(self, element_time_steps):
        """Decodes list of years to base time steps.

        :param element_time_steps: time steps of year
        :return: full_base_time_steps: full list of time steps
        """
        list_base_time_steps = []
        for year in element_time_steps:
            list_base_time_steps.append(self.decode_time_step(year, "yearly"))
        full_base_time_steps = np.concatenate(list_base_time_steps)
        return full_base_time_steps

    def convert_time_step_energy2power(self, time_step_energy):
        """Converts energy time steps to power time steps of storage technologies.

        converts the time step of the energy quantities of a storage technology
        to the time step of the power quantities.

        :param time_step_energy: time step of energy quantities
        :return: time step of power quantities
        """
        assert self.time_steps_energy2power is not None
        time_steps_energy2power = self.time_steps_energy2power
        return time_steps_energy2power[time_step_energy]

    def convert_time_step_operation2year(self, time_step_operation):
        """Converts the operational time step to the invest time step.

        :param time_step_operation: time step of operational time steps
        :return: time step of invest time steps
        """
        assert self.time_steps_operation2year is not None
        time_steps_operation2year = self.time_steps_operation2year
        return time_steps_operation2year[time_step_operation]

    def initialize_representative_hours(self, years: list[int], system) -> None:
        """Build the year-aware time representation used by the optimization model.

        The aggregation pipeline still creates globally unique time-step identifiers.
        This method converts them into explicit ``(year, representative_hour)`` pairs.
        The flat identifiers remain available only for result serialization.

        Args:
            years: Complete ordered list of modeled years.
            system: System configuration controlling storage periodicity.

        Raises:
            ValueError: If different years contain different numbers of operational
                representative hours.
        """
        assert self.time_steps_year2operation is not None
        assert self.time_steps_year2storage is not None
        assert self.time_steps_operation_duration is not None
        assert self.time_steps_storage_duration is not None
        assert self.time_steps_energy2power is not None

        operation_steps_by_year = {
            year: [int(step) for step in self.time_steps_year2operation[year]]
            for year in years
        }
        counts = {year: len(steps) for year, steps in operation_steps_by_year.items()}
        expected_count = counts[years[0]]
        unequal = {
            year: count for year, count in counts.items() if count != expected_count
        }
        if unequal:
            details = ", ".join(
                f"year {year}: {count}" for year, count in unequal.items()
            )
            raise ValueError(
                "Year-specific time-series aggregation must return the same number "
                f"of representative hours for every year. Expected {expected_count}; "
                f"received {details}."
            )

        self.representative_hours = list(range(expected_count))
        self.operation_step_to_pair = {
            step: (year, representative_hour)
            for year, steps in operation_steps_by_year.items()
            for representative_hour, step in enumerate(steps)
        }
        self.operation_pair_to_step = {
            pair: step for step, pair in self.operation_step_to_pair.items()
        }
        operation_index = pd.MultiIndex.from_tuples(
            list(self.operation_pair_to_step),
            names=["set_years", "set_representative_hours"],
        )
        self.weight_representative_hours = pd.Series(
            [
                self.time_steps_operation_duration[self.operation_pair_to_step[pair]]
                for pair in operation_index
            ],
            index=operation_index,
            dtype=int,
        )

        storage_steps_by_year = {
            year: [int(step) for step in self.time_steps_year2storage[year]]
            for year in years
        }
        self.representative_hours_storage = {
            year: list(range(len(steps)))
            for year, steps in storage_steps_by_year.items()
        }
        self.storage_step_to_pair = {
            step: (year, representative_hour)
            for year, steps in storage_steps_by_year.items()
            for representative_hour, step in enumerate(steps)
        }
        self.storage_pair_to_step = {
            pair: step for step, pair in self.storage_step_to_pair.items()
        }
        storage_index = pd.MultiIndex.from_tuples(
            list(self.storage_pair_to_step),
            names=["set_years", "set_representative_hours_storage"],
        )
        self.duration_representative_hours_storage = pd.Series(
            [
                self.time_steps_storage_duration[self.storage_pair_to_step[pair]]
                for pair in storage_index
            ],
            index=storage_index,
            dtype=int,
        )
        self.storage_to_operation = pd.Series(
            [
                self.operation_step_to_pair[
                    self.time_steps_energy2power[self.storage_pair_to_step[pair]]
                ][1]
                for pair in storage_index
            ],
            index=storage_index,
            name="set_representative_hours",
            dtype=int,
        )
        self._set_storage_predecessors(storage_steps_by_year, years, system)

    def _set_storage_predecessors(
        self,
        storage_steps_by_year: dict[int, list[int]],
        years: list[int],
        system,
    ) -> None:
        """Create predecessor pairs and coupling flags for storage states."""
        assert self.storage_step_to_pair is not None
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        coupling: dict[tuple[int, int], bool] = {}
        for year_position, year in enumerate(years):
            steps = storage_steps_by_year[year]
            for position, step in enumerate(steps):
                pair = self.storage_step_to_pair[step]
                if position > 0:
                    predecessor_step = steps[position - 1]
                    is_coupled = True
                elif system.multiyear_periodicity and year_position > 0:
                    predecessor_step = storage_steps_by_year[years[year_position - 1]][
                        -1
                    ]
                    is_coupled = True
                elif system.multiyear_periodicity and system.storage_periodicity:
                    predecessor_step = storage_steps_by_year[years[-1]][-1]
                    is_coupled = True
                elif system.storage_periodicity:
                    predecessor_step = steps[-1]
                    is_coupled = True
                else:
                    predecessor_step = step
                    is_coupled = False
                previous[pair] = self.storage_step_to_pair[predecessor_step]
                coupling[pair] = is_coupled

        index = pd.MultiIndex.from_tuples(
            list(previous),
            names=["set_years", "set_representative_hours_storage"],
        )
        self.previous_storage_time_step = pd.Series(
            [previous[pair] for pair in index], index=index, dtype=object
        )
        self.storage_coupling_mask = pd.Series(
            [coupling[pair] for pair in index], index=index, dtype=bool
        )
