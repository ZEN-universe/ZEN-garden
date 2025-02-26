"""
This file implements a helper class to deal with timesteps
"""

import numpy as np
import pandas as pd
import logging


class TimeStepsDicts(object):
    """
    This class implements some simple helper functions that can deal with the time steps of the optimization setup.
    It is very similar to EnergySystem functions and is meant to avoid the import of packages that can cause conflicts.
    """

    def __init__(self, dict_all_sequence_time_steps=None):
        """
        Sets all dicts of sequences of time steps.

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

        # set the params if provided
        else:
            el_op = [el for el in dict_all_sequence_time_steps["operation"].keys() if "storage_level" not in el][0]
            el_stor = [el for el in dict_all_sequence_time_steps["operation"].keys() if "storage_level" in el][0]
            self.sequence_time_steps_operation = dict_all_sequence_time_steps["operation"][el_op]
            self.sequence_time_steps_storage = dict_all_sequence_time_steps["operation"][el_stor]
            self.sequence_time_steps_yearly = dict_all_sequence_time_steps["yearly"][None]
            self.set_time_steps_operation2year_both_dir()
            self.set_time_steps_storage2year_both_dir()

    def set_aggregated_time_steps(self,tsa):
        """
        sets the aggregated time steps types

        :param tsa: time series aggregation object
        """
        self.time_steps_operation = list(tsa.set_time_steps)
        self.time_steps_operation_duration = tsa.time_steps_duration
        self.sequence_time_steps_operation = tsa.sequence_time_steps

    def get_sequence_time_steps(self, time_step_type="operation"):
        """
        Get sequence ot time steps of element

        :param time_step_type: type of time step (operation, storage or yearly)
        :return sequence_time_steps: list of time steps corresponding to base time step
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
        """
        Returns all dicts of sequence of time steps.

        :return dict_all_sequence_time_steps: dict of all dict_sequence_time_steps
        """

        dict_all_sequence_time_steps = {"operation": self.sequence_time_steps_operation,
                                        "storage": self.sequence_time_steps_storage,
                                        "yearly": self.sequence_time_steps_yearly}
        return dict_all_sequence_time_steps

    def encode_time_step(self, base_time_steps: int, time_step_type: str = None):
        """
        Encodes baseTimeStep, i.e., retrieves the time step of an element corresponding to baseTimeStep of model.
        baseTimeStep of model --> timeStep of element

        :param base_time_steps: base time step of model for which the corresponding time index is extracted
        :param time_step_type: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element
        """
        sequence_time_steps = self.get_sequence_time_steps(time_step_type)
        # get time step duration
        if np.all(base_time_steps >= 0):
            element_time_step = np.unique(sequence_time_steps[base_time_steps])
        else:
            element_time_step = [-1]
        return (element_time_step)

    def decode_time_step(self, element_time_step: int, time_step_type: str = None):
        """
        Decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the variableTimeStep of a element.
        timeStep of element --> baseTimeStep of model

        :param element: element of model, i.e., carrier or technology
        :param element_time_step: time step of element
        :param time_step_type: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model
        """
        sequence_time_steps = self.get_sequence_time_steps(time_step_type)
        # find where element_time_step in sequence of element time steps
        base_time_steps = np.argwhere(sequence_time_steps == element_time_step)
        return base_time_steps

    def calculate_time_step_duration(self, input_time_steps, base_time_steps):
        """ calculates (equidistant) time step durations for input time steps

        :param input_time_steps: input time steps
        :param base_time_steps: manual list of base time steps
        :return time_step_duration_dict: dict with duration of each time step """
        duration_input_time_steps = len(base_time_steps) / len(input_time_steps)
        time_step_duration_dict = {time_step: int(duration_input_time_steps) for time_step in input_time_steps}
        if not duration_input_time_steps.is_integer():
            logging.warning(f"The duration of each time step {duration_input_time_steps} of input time steps {input_time_steps} does not evaluate to an integer. \n"
                            f"The duration of the last time step is set to compensate for the difference")
            duration_last_time_step = len(base_time_steps) - sum(time_step_duration_dict[key] for key in time_step_duration_dict if key != input_time_steps[-1])
            time_step_duration_dict[input_time_steps[-1]] = duration_last_time_step
        return time_step_duration_dict

    def set_time_steps_operation2year_both_dir(self):
        """ calculates the conversion of operational time steps to invest/yearly time steps

        """
        sequence_operation = self.sequence_time_steps_operation
        sequence_yearly = self.sequence_time_steps_yearly
        time_steps_combi_operation = np.vstack(pd.unique(pd.Series(zip(sequence_operation, sequence_yearly)))).T
        time_steps_combi_operation = np.sort(time_steps_combi_operation,axis=1)
        # calculate operation2year
        time_steps_operation2year = {key: val for key, val in zip(time_steps_combi_operation[0, :], time_steps_combi_operation[1, :])}
        self.time_steps_operation2year = time_steps_operation2year
        # calculate year2operation
        time_steps_year2operation = {}
        for year in pd.unique(time_steps_combi_operation[1]):
            time_steps_year2operation[year] = time_steps_combi_operation[0,time_steps_combi_operation[1] == year]
        self.time_steps_year2operation = time_steps_year2operation

    def set_time_steps_storage2year_both_dir(self):
        """ calculates the conversion of storage time steps to invest/yearly time steps

        """
        sequence_storage = self.sequence_time_steps_storage
        sequence_yearly = self.sequence_time_steps_yearly
        time_steps_combi_storage = np.vstack(pd.unique(pd.Series(zip(sequence_storage, sequence_yearly)))).T
        # calculate storage2year
        time_steps_storage2year = {key: val for key, val in zip(time_steps_combi_storage[0, :], time_steps_combi_storage[1, :])}
        self.time_steps_storage2year = time_steps_storage2year
        # calculate year2storage
        time_steps_year2storage = {}
        for year in pd.unique(time_steps_combi_storage[1]):
            time_steps_year2storage[year] = time_steps_combi_storage[0,time_steps_combi_storage[1] == year]
        self.time_steps_year2storage = time_steps_year2storage

    def set_time_steps_storage_startend(self, system):
        """ sets the dict of matching the last time step of the year in the storage level domain to the first

        :param system: dictionary defining the system
        """
        unaggregated_time_steps = system.unaggregated_time_steps_per_year
        sequence_time_steps = self.sequence_time_steps_storage
        counter = 0
        time_steps_start = []
        time_steps_end = []
        assert system.interval_between_years == 1 or not system.multiyear_periodicity, "The interval between years should be 1 for multiyear storage periodicity."
        if not system.multiyear_periodicity:
            while counter < len(sequence_time_steps):
                time_steps_start.append(sequence_time_steps[counter])
                counter += unaggregated_time_steps
                time_steps_end.append(sequence_time_steps[counter - 1])
            self.time_steps_storage_level_startend_year = {start: end for start, end in zip(time_steps_start, time_steps_end)}
        else:
            self.time_steps_storage_level_startend_year = {self.sequence_time_steps_storage[0]: self.sequence_time_steps_storage[-1]}

    def get_time_steps_year2operation(self, year=None):
        """ gets the dict of converting the invest time steps to the operation time steps of technologies

        :param year: year of interest
        :return: time_steps_year2operation of the specified element (at specified year)
        """
        if year is None:
            return self.time_steps_year2operation
        else:
            return self.time_steps_year2operation[year]

    def get_time_steps_storage2year(self):
        """ gets the dict of converting the storage time steps to the invest time steps of technologies

        :return: time_steps_storage2year of the specified element
        """
        return self.time_steps_storage2year

    def get_time_steps_year2storage(self, year=None):
        """ gets the dict of converting the invest time steps to the storage time steps of technologies

        :param year: year of interest
        :return: time_steps_year2storage of the specified element (at specified year)
        """
        if year is None:
            return self.time_steps_year2storage
        else:
            return self.time_steps_year2storage[year]

    def get_time_steps_storage_startend(self, time_step):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies

        :param time_step: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        if time_step in self.time_steps_storage_level_startend_year.keys():
            return self.time_steps_storage_level_startend_year[time_step]
        else:
            return None

    def get_previous_storage_time_step(self, time_step):
        """ gets the storage time step before in the sequence

        :param time_step: current time step
        :return previous_time_step: previous time step
        """
        sequence = self.sequence_time_steps_storage
        previous_time_step = sequence[np.where(sequence == time_step)[0] - 1][0]
        return previous_time_step

    def decode_yearly_time_steps(self, element_time_steps):
        """ decodes list of years to base time steps

        :param element_time_steps: time steps of year
        :return full_base_time_steps: full list of time steps """
        list_base_time_steps = []
        for year in element_time_steps:
            list_base_time_steps.append(self.decode_time_step(year, "yearly"))
        full_base_time_steps = np.concatenate(list_base_time_steps)
        return full_base_time_steps

    def convert_time_step_energy2power(self, time_step_energy):
        """ converts the time step of the energy quantities of a storage technology to the time step of the power quantities

        :param time_step_energy: time step of energy quantities
        :return: time step of power quantities
        """
        time_steps_energy2power = self.time_steps_energy2power
        return time_steps_energy2power[time_step_energy]

    def convert_time_step_operation2year(self, time_step_operation):
        """ converts the operational time step to the invest time step

        :param time_step_operation: time step of operational time steps
        :return: time step of invest time steps
        """
        time_steps_operation2year = self.time_steps_operation2year
        return time_steps_operation2year[time_step_operation]
