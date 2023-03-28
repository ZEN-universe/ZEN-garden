"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  This file implements a helper class to deal with timesteps
==========================================================================================================================================================================="""

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

        # empty dict of sequence of time steps operation
        self.dict_sequence_time_steps_operation = {}
        # empty dict of sequence of time steps yearly
        self.dict_sequence_time_steps_yearly = {}
        # empty dict of conversion from energy time steps to power time steps for storage technologies
        self.dict_time_steps_energy2power = {}
        # empty dict of conversion from operational time steps to invest time steps for technologies
        self.dict_time_steps_operation2year = {}
        # empty dict of conversion from invest time steps to operation time steps for technologies
        self.dict_time_steps_year2operation = {}
        # empty dict of matching the last time step of the year in the storage domain to the first
        self.dict_time_steps_storage_level_startend_year = {}

        # set the params if provided
        if dict_all_sequence_time_steps is not None:
            self.reset_dicts(dict_all_sequence_time_steps)
        else:
            self.dict_sequence_time_steps_operation = dict()
            self.dict_sequence_time_steps_yearly = dict()

    def set_sequence_time_steps(self, element, sequence_time_steps, time_step_type=None):
        """
        Sets sequence of time steps, either of operation, invest, or year
        :param element: name of element in model
        :param sequence_time_steps: list of time steps corresponding to base time step
        :param time_step_type: type of time step (operation or yearly)
        """

        if not time_step_type:
            time_step_type = "operation"
        if time_step_type == "operation":
            self.dict_sequence_time_steps_operation[element] = pd.Series(sequence_time_steps)
        elif time_step_type == "yearly":
            self.dict_sequence_time_steps_yearly[element] = pd.Series(sequence_time_steps)
        else:
            raise KeyError(f"Time step type {time_step_type} is incorrect")

    def reset_dicts(self, dict_all_sequence_time_steps):
        """
        Resets all dicts of sequences of time steps.
        :param dict_all_sequence_time_steps: dict of all dict_sequence_time_steps
        """

        for k, v in dict_all_sequence_time_steps["operation"].items():
            self.set_sequence_time_steps(k, v, time_step_type="operation")
        for k, v in dict_all_sequence_time_steps["yearly"].items():
            self.set_sequence_time_steps(k, v, time_step_type="yearly")

    def get_sequence_time_steps(self, element, time_step_type=None):
        """
        Get sequence ot time steps of element
        :param element: name of element in model
        :param time_step_type: type of time step (operation or invest)
        :return sequence_time_steps: list of time steps corresponding to base time step
        """

        if not time_step_type:
            time_step_type = "operation"
        if time_step_type == "operation":
            return self.dict_sequence_time_steps_operation[element]
        elif time_step_type == "yearly":
            return self.dict_sequence_time_steps_yearly[None]
        else:
            raise KeyError(f"Time step type {time_step_type} is incorrect")

    def get_sequence_time_steps_dict(self):
        """
        Returns all dicts of sequence of time steps.
        :return dict_all_sequence_time_steps: dict of all dict_sequence_time_steps
        """

        dict_all_sequence_time_steps = {"operation": self.dict_sequence_time_steps_operation,
                                        "yearly": self.dict_sequence_time_steps_yearly}
        return dict_all_sequence_time_steps

    def encode_time_step(self, element: str, base_time_steps: int, time_step_type: str = None, yearly=False):
        """
        Encodes baseTimeStep, i.e., retrieves the time step of a element corresponding to baseTimeStep of model.
        baseTimeStep of model --> timeStep of element
        :param element: name of element in model, i.e., carrier or technology
        :param base_time_steps: base time step of model for which the corresponding time index is extracted
        :param time_step_type: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element
        """
        sequence_time_steps = self.get_sequence_time_steps(element, time_step_type)
        # get time step duration
        if np.all(base_time_steps >= 0):
            element_time_step = np.unique(sequence_time_steps[base_time_steps])
        else:
            element_time_step = [-1]
        if yearly:
            return (element_time_step)
        if len(element_time_step) == 1:
            return (element_time_step[0])
        else:
            raise LookupError(f"Currently only implemented for a single element time step, not {element_time_step}")

    def decode_time_step(self, element, element_time_step: int, time_step_type: str = None):
        """
        Decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the variableTimeStep of a element.
        timeStep of element --> baseTimeStep of model
        :param element: element of model, i.e., carrier or technology
        :param element_time_step: time step of element
        :param time_step_type: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model
        """
        sequence_time_steps = self.get_sequence_time_steps(element, time_step_type)
        # find where element_time_step in sequence of element time steps
        base_time_steps = sequence_time_steps[sequence_time_steps == element_time_step].values
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

    def set_time_steps_energy2power(self, element, time_steps_energy2power):
        """ sets the dict of converting the energy time steps to the power time steps of storage technologies """
        self.dict_time_steps_energy2power[element] = pd.Series(time_steps_energy2power)

    def set_time_steps_operation2year_both_dir(self,element_name,sequence_operation,sequence_yearly):
        """ calculates the conversion of operational time steps to invest/yearly time steps """
        # time_steps_combi = np.unique(np.vstack([sequence_operation, sequence_yearly]), axis=1)
        time_steps_combi = np.vstack(pd.unique(list(zip(sequence_operation, sequence_yearly)))).T
        time_steps_operation2year = {key: val for key, val in zip(time_steps_combi[0, :], time_steps_combi[1, :])}
        self.set_time_steps_operation2year(element_name, time_steps_operation2year)
        # calculate year2operation
        time_steps_year2operation = {}
        for year in pd.unique(time_steps_combi[1]):
            time_steps_year2operation[year] = time_steps_combi[0,time_steps_combi[1] == year]
        self.set_time_steps_year2operation(element_name, time_steps_year2operation)

    def set_time_steps_operation2year(self, element, time_steps_operation2year):
        """ sets the dict of converting the operational time steps to the invest time steps of all elements """
        self.dict_time_steps_operation2year[element] = pd.Series(time_steps_operation2year)

    def set_time_steps_year2operation(self, element, time_steps_year2operation):
        """ sets the dict of converting the operational time steps to the invest time steps of all elements """
        self.dict_time_steps_year2operation[element] = pd.Series(time_steps_year2operation)

    def set_time_steps_storage_startend(self, element, system):
        """ sets the dict of matching the last time step of the year in the storage level domain to the first """
        _unaggregated_time_steps = system["unaggregated_time_steps_per_year"]
        _sequence_time_steps = self.get_sequence_time_steps(element + "_storage_level")
        _counter = 0
        _time_steps_start = []
        _time_steps_end = []
        while _counter < len(_sequence_time_steps):
            _time_steps_start.append(_sequence_time_steps[_counter])
            _counter += _unaggregated_time_steps
            _time_steps_end.append(_sequence_time_steps[_counter - 1])
        self.dict_time_steps_storage_level_startend_year[element] = pd.Series({_start: _end for _start, _end in zip(_time_steps_start, _time_steps_end)})

    def set_sequence_time_steps_dict(self, dict_all_sequence_time_steps):
        """ sets all dicts of sequences of time steps.
        :param dict_all_sequence_time_steps: dict of all dict_sequence_time_steps"""
        self.reset_dicts(dict_all_sequence_time_steps=dict_all_sequence_time_steps)

    def get_time_steps_energy2power(self, element):
        """ gets the dict of converting the energy time steps to the power time steps of storage technologies """
        return self.dict_time_steps_energy2power[element]

    def get_time_steps_operation2year(self, element):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        return self.dict_time_steps_operation2year[element]

    def get_time_steps_year2operation(self, element,year=None):
        """ gets the dict of converting the invest time steps to the operation time steps of technologies """
        if year is None:
            return self.dict_time_steps_year2operation[element]
        else:
            return self.dict_time_steps_year2operation[element][year]

    def get_time_steps_storage_startend(self, element, time_step):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        if time_step in self.dict_time_steps_storage_level_startend_year[element].keys():
            return self.dict_time_steps_storage_level_startend_year[element][time_step]
        else:
            return None

    def decode_yearly_time_steps(self, element_time_steps):
        """ decodes list of years to base time steps
        :param element_time_steps: time steps of year
        :return _full_base_time_steps: full list of time steps """
        _list_base_time_steps = []
        for year in element_time_steps:
            _list_base_time_steps.append(self.decode_time_step(None, year, "yearly"))
        _full_base_time_steps = np.concatenate(_list_base_time_steps)
        return _full_base_time_steps

    def convert_time_step_energy2power(self, element, _time_step_energy):
        """ converts the time step of the energy quantities of a storage technology to the time step of the power quantities """
        _time_steps_energy2power = self.get_time_steps_energy2power(element)
        return _time_steps_energy2power[_time_step_energy]

    def convert_time_step_operation2year(self, element, time_step_operation):
        """ converts the operational time step to the invest time step """
        time_steps_operation2year = self.get_time_steps_operation2year(element)
        return time_steps_operation2year[time_step_operation]
