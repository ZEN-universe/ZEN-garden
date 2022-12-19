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

    def __init__(self, dictAllSequenceTimeSteps=None):
        """
        Sets all dicts of sequences of time steps.
        :param dictAllSequenceTimeSteps: dict of all dictSequenceTimeSteps
        """

        # set the params if provided
        if dictAllSequenceTimeSteps is not None:
            self.dictSequenceTimeStepsOperation = dictAllSequenceTimeSteps["operation"]
            self.dictSequenceTimeStepsYearly = dictAllSequenceTimeSteps["yearly"]
        else:
            self.dictSequenceTimeStepsOperation = dict()
            self.dictSequenceTimeStepsYearly = dict()

    def setSequenceTimeSteps(self, element, sequenceTimeSteps, timeStepType=None):
        """
        Sets sequence of time steps, either of operation, invest, or year
        :param element: name of element in model
        :param sequenceTimeSteps: list of time steps corresponding to base time step
        :param timeStepType: type of time step (operation or yearly)
        """

        if not timeStepType:
            timeStepType = "operation"
        if timeStepType == "operation":
            self.dictSequenceTimeStepsOperation[element] = sequenceTimeSteps
        elif timeStepType == "yearly":
            self.dictSequenceTimeStepsYearly[element] = sequenceTimeSteps
        else:
            raise KeyError(f"Time step type {timeStepType} is incorrect")

    def reset_dicts(self, dictAllSequenceTimeSteps):
        """
        Resets all dicts of sequences of time steps.
        :param dictAllSequenceTimeSteps: dict of all dictSequenceTimeSteps
        """

        self.dictSequenceTimeStepsOperation = dictAllSequenceTimeSteps["operation"]
        self.dictSequenceTimeStepsYearly = dictAllSequenceTimeSteps["yearly"]

    def getSequenceTimeSteps(self, element, timeStepType=None):
        """
        Get sequence ot time steps of element
        :param element: name of element in model
        :param timeStepType: type of time step (operation or invest)
        :return sequenceTimeSteps: list of time steps corresponding to base time step
        """

        if not timeStepType:
            timeStepType = "operation"
        if timeStepType == "operation":
            return self.dictSequenceTimeStepsOperation[element]
        elif timeStepType == "yearly":
            return self.dictSequenceTimeStepsYearly[None]
        else:
            raise KeyError(f"Time step type {timeStepType} is incorrect")

    def getSequenceTimeStepsDict(self):
        """
        Returns all dicts of sequence of time steps.
        :return dictAllSequenceTimeSteps: dict of all dictSequenceTimeSteps
        """

        dictAllSequenceTimeSteps = {
            "operation": self.dictSequenceTimeStepsOperation,
            "yearly": self.dictSequenceTimeStepsYearly
        }
        return dictAllSequenceTimeSteps

    def encodeTimeStep(cls, element: str, baseTimeSteps: int, timeStepType: str = None, yearly=False):
        """
        Encodes baseTimeStep, i.e., retrieves the time step of a element corresponding to baseTimeStep of model.
        baseTimeStep of model --> timeStep of element
        :param element: name of element in model, i.e., carrier or technology
        :param baseTimeStep: base time step of model for which the corresponding time index is extracted
        :param timeStepType: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element
        """
        sequenceTimeSteps = cls.getSequenceTimeSteps(element, timeStepType)
        # get time step duration
        if np.all(baseTimeSteps >= 0):
            elementTimeStep = np.unique(sequenceTimeSteps[baseTimeSteps])
        else:
            elementTimeStep = [-1]
        if yearly:
            return (elementTimeStep)
        if len(elementTimeStep) == 1:
            return (elementTimeStep[0])
        else:
            raise LookupError(f"Currently only implemented for a single element time step, not {elementTimeStep}")

    def decodeTimeStep(self, element, elementTimeStep:int, timeStepType:str = None):
        """
        Decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the variableTimeStep of a element.
        timeStep of element --> baseTimeStep of model
        :param element: element of model, i.e., carrier or technology
        :param elementTimeStep: time step of element
        :param timeStepType: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model
        """
        sequenceTimeSteps = self.getSequenceTimeSteps(element,timeStepType)
        # find where elementTimeStep in sequence of element time steps
        baseTimeSteps = np.argwhere(sequenceTimeSteps == elementTimeStep)
        return baseTimeSteps

