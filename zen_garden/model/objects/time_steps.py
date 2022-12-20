"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  This file implements a helper class to deal with timesteps
==========================================================================================================================================================================="""

import numpy as np


class SequenceTimeStepsDicts(object):
    """
    This class implements some simple helper functions that can deal with the time steps of the optimization setup.
    It is very similar to EnergySystem functions and is meant to avoid the import of packages that can cause conflicts.
    """

    def __init__(self, dict_all_sequence_time_steps=None):
        """
        Sets all dicts of sequences of time steps.
        :param dict_all_sequence_time_steps: dict of all dict_sequence_time_steps
        """

        # set the params if provided
        if dict_all_sequence_time_steps is not None:
            self.dict_sequence_time_steps_operation = dict_all_sequence_time_steps["operation"]
            self.dict_sequence_time_steps_yearly = dict_all_sequence_time_steps["yearly"]
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
            self.dict_sequence_time_steps_operation[element] = sequence_time_steps
        elif time_step_type == "yearly":
            self.dict_sequence_time_steps_yearly[element] = sequence_time_steps
        else:
            raise KeyError(f"Time step type {time_step_type} is incorrect")

    def reset_dicts(self, dict_all_sequence_time_steps):
        """
        Resets all dicts of sequences of time steps.
        :param dict_all_sequence_time_steps: dict of all dict_sequence_time_steps
        """

        self.dict_sequence_time_steps_operation = dict_all_sequence_time_steps["operation"]
        self.dict_sequence_time_steps_yearly = dict_all_sequence_time_steps["yearly"]

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

        dict_all_sequence_time_steps = {
            "operation": self.dict_sequence_time_steps_operation,
            "yearly": self.dict_sequence_time_steps_yearly
        }
        return dict_all_sequence_time_steps

    def encode_time_step(cls, element: str, base_time_steps: int, time_step_type: str = None, yearly=False):
        """
        Encodes baseTimeStep, i.e., retrieves the time step of a element corresponding to baseTimeStep of model.
        baseTimeStep of model --> timeStep of element
        :param element: name of element in model, i.e., carrier or technology
        :param baseTimeStep: base time step of model for which the corresponding time index is extracted
        :param time_step_type: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element
        """
        sequence_time_steps = cls.get_sequence_time_steps(element, time_step_type)
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

    def decode_time_step(self, element, element_time_step:int, time_step_type:str = None):
        """
        Decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the variableTimeStep of a element.
        timeStep of element --> baseTimeStep of model
        :param element: element of model, i.e., carrier or technology
        :param element_time_step: time step of element
        :param time_step_type: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model
        """
        sequence_time_steps = self.get_sequence_time_steps(element,time_step_type)
        # find where element_time_step in sequence of element time steps
        base_time_steps = np.argwhere(sequence_time_steps == element_time_step)
        return base_time_steps

