"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Functions to apply time series aggregation to time series
==========================================================================================================================================================================="""
import cProfile

import pandas as pd
import numpy as np
import logging
import tsam.timeseriesaggregation as tsam
import pstats
from zen_garden.model.objects.energy_system import EnergySystem
from zen_garden.model.objects.element import Element
from zen_garden.model.objects.carrier import Carrier
from zen_garden.model.objects.technology.technology import Technology
from zen_garden.model.objects.technology.storage_technology import StorageTechnology

class TimeSeriesAggregation():
    timeSeriesAggregation = None

    def __init__(self):
        """ initializes the time series aggregation. The data is aggregated for a single year and then concatenated"""
        self.system                 = EnergySystem.get_system()
        self.analysis               = EnergySystem.get_analysis()
        self.energy_system           = EnergySystem.get_energy_system()
        self.headerSetTimeSteps     = self.analysis['headerDataInputs']["set_time_steps"]
        # if set_time_steps as input (because already aggregated), use this as base time step, otherwise self.set_base_time_steps
        self.set_base_time_steps       = self.energy_system.set_base_time_steps_yearly
        self.numberTypicalPeriods   = min(self.system["unaggregatedTimeStepsPerYear"], self.system["aggregatedTimeStepsPerYear"])
        # set timeSeriesAggregation
        TimeSeriesAggregation.setTimeSeriesAggregation(self)
        self.conductedTimeSeriesAggregation = False
        # if number of time steps >= number of base time steps, skip aggregation
        if self.numberTypicalPeriods < np.size(self.set_base_time_steps) and self.system["conductTimeSeriesAggregation"]:
            # select time series
            self.selectTimeSeriesOfAllElements() #TODO speed up
            if not self.dfTimeSeriesRaw.empty:
                # run time series aggregation to create typical periods
                self.runTimeSeriesAggregation()
        else:
            self.typicalPeriods = pd.DataFrame()
            _setTimeSteps       = self.set_base_time_steps
            _timeStepDuration   = EnergySystem.calculate_time_step_duration(_setTimeSteps,self.set_base_time_steps)
            _sequence_time_steps  = np.concatenate([[time_step] * _timeStepDuration[time_step] for time_step in _timeStepDuration])
            TimeSeriesAggregation.setTimeAttributes(self,_setTimeSteps,_timeStepDuration,_sequence_time_steps)
            # set aggregated time series
            self.setAggregatedTimeSeriesOfAllElements()

    def selectTimeSeriesOfAllElements(self):
        """ this method retrieves the raw time series for the aggregation of all input data sets.
        Only in aligned time grid approach! """

        _dictRawTimeSeries = {}
        for element in Element.get_all_elements():
            dfTimeSeriesRaw = TimeSeriesAggregation.extractRawTimeSeries(element, self.headerSetTimeSteps)
            if not dfTimeSeriesRaw.empty:
                _dictRawTimeSeries[element.name] = dfTimeSeriesRaw
        if _dictRawTimeSeries:
            self.dfTimeSeriesRaw = pd.concat(_dictRawTimeSeries.values(), axis=1, keys=_dictRawTimeSeries.keys())
        else:
            self.dfTimeSeriesRaw = pd.Series()

    def substituteColumnNames(self, direction="flatten"):
        """ this method substitutes the column names to have flat columns names (otherwise sklearn warning) """
        if direction == "flatten":
            if not hasattr(self,"columnNamesOriginal"):
                self.columnNamesOriginal        = self.dfTimeSeriesRaw.columns
                self.columnNamesFlat            = [str(index) for index in self.columnNamesOriginal]
                self.dfTimeSeriesRaw.columns    = self.columnNamesFlat
        elif direction == "raise":
            self.typicalPeriods             = self.typicalPeriods[self.columnNamesFlat]
            self.typicalPeriods.columns     = self.columnNamesOriginal

    def runTimeSeriesAggregation(self):
        """ this method runs the time series aggregation """
        # substitute column names
        self.substituteColumnNames(direction="flatten")
        # create aggregation object
        self.aggregation = tsam.TimeSeriesAggregation(
            timeSeries           = self.dfTimeSeriesRaw,
            noTypicalPeriods     = self.numberTypicalPeriods,
            hoursPerPeriod       = self.analysis["timeSeriesAggregation"]["hoursPerPeriod"],
            resolution           = self.analysis["timeSeriesAggregation"]["resolution"],
            clusterMethod        = self.analysis["timeSeriesAggregation"]["clusterMethod"],
            solver               = self.analysis["timeSeriesAggregation"]["solver"],
            extremePeriodMethod  = self.analysis["timeSeriesAggregation"]["extremePeriodMethod"],
            rescaleClusterPeriods= self.analysis["timeSeriesAggregation"]["rescaleClusterPeriods"],
            representationMethod = self.analysis["timeSeriesAggregation"]["representationMethod"]
        )
        # create typical periods
        self.typicalPeriods     = self.aggregation.createTypicalPeriods().reset_index(drop=True)
        TimeSeriesAggregation.setTimeAttributes(self,self.aggregation.clusterPeriodIdx,self.aggregation.clusterPeriodNoOccur,self.aggregation.clusterOrder)
        # resubstitute column names
        self.substituteColumnNames(direction="raise")
        # set aggregated time series
        self.setAggregatedTimeSeriesOfAllElements()
        self.conductedTimeSeriesAggregation = True

    def setAggregatedTimeSeriesOfAllElements(self):
        """ this method sets the aggregated time series and sets the necessary attributes after the aggregation to a single time grid """
        for element in Element.get_all_elements():
            raw_time_series = getattr(element, "raw_time_series")
            # set_time_steps, duration and sequence time steps
            element.setTimeStepsOperation       = list(self.set_time_steps)
            element.timeStepsOperationDuration  = self.timeStepsDuration
            element.sequence_time_steps           = self.sequence_time_steps

            # iterate through raw time series
            for timeSeries in raw_time_series:
                _index_names = list(raw_time_series[timeSeries].index.names)
                _index_names.remove(self.headerSetTimeSteps)
                dfTimeSeries = raw_time_series[timeSeries].unstack(level=_index_names)

                dfAggregatedTimeSeries = pd.DataFrame(index=self.set_time_steps, columns=dfTimeSeries.columns)
                # columns which are in aggregated time series and which are not
                if element.name in self.typicalPeriods and timeSeries in self.typicalPeriods[element.name]:
                    dfTypicalPeriods = self.typicalPeriods[element.name, timeSeries]
                    AggregatedColumns = dfTimeSeries.columns.intersection(dfTypicalPeriods.columns)
                    NotAggregatedColumns = dfTimeSeries.columns.difference(dfTypicalPeriods.columns)
                    # aggregated columns
                    dfAggregatedTimeSeries[AggregatedColumns] = self.typicalPeriods[element.name, timeSeries][
                        AggregatedColumns]
                else:
                    NotAggregatedColumns = dfTimeSeries.columns
                # not aggregated columns
                dfAggregatedTimeSeries[NotAggregatedColumns] = dfTimeSeries.loc[
                    dfAggregatedTimeSeries.index, NotAggregatedColumns]
                # reorder
                dfAggregatedTimeSeries.index.names      = [self.headerSetTimeSteps]
                dfAggregatedTimeSeries.columns.names    = _index_names
                dfAggregatedTimeSeries                  = dfAggregatedTimeSeries.stack(_index_names)
                dfAggregatedTimeSeries.index            = dfAggregatedTimeSeries.index.reorder_levels(
                                                            _index_names + [self.headerSetTimeSteps])
                setattr(element, timeSeries, dfAggregatedTimeSeries)
                TimeSeriesAggregation.setAggregationIndicators(element)

    @classmethod
    def extractRawTimeSeries(cls, element, headerSetTimeSteps):
        """ extract the time series from an element and concatenates the non-constant time series to a pd.DataFrame
        :param element: element of the optimization 
        :param headerSetTimeSteps: name of set_time_steps
        :return dfTimeSeriesRaw: pd.DataFrame with non-constant time series"""
        _dictRawTimeSeries = {}
        raw_time_series = getattr(element, "raw_time_series")
        for timeSeries in raw_time_series:
            raw_time_series[timeSeries].name = timeSeries
            _index_names = list(raw_time_series[timeSeries].index.names)
            _index_names.remove(headerSetTimeSteps)
            dfTimeSeries = raw_time_series[timeSeries].unstack(level=_index_names)
            # select time series that are not constant (rows have more than 1 unique entries)
            dfTimeSeriesNonConstant = dfTimeSeries[
                dfTimeSeries.columns[dfTimeSeries.apply(lambda column: len(np.unique(column)) != 1)]]
            _dictRawTimeSeries[timeSeries] = dfTimeSeriesNonConstant
        dfTimeSeriesRaw = pd.concat(_dictRawTimeSeries.values(), axis=1, keys=_dictRawTimeSeries.keys())
        return dfTimeSeriesRaw

    @classmethod
    def calculateTimeStepsEnergyBalance(cls, element):
        """ calculates the necessary time steps of carrier. Carrier must always have highest resolution of all connected technologies. 
        Can have higher resolution
        :param element: element of the optimization """
        # if carrier is already aggregated
        if element.isAggregated():
            sequenceTimeStepsRaw = element.sequence_time_steps
            # if sequenceTimeStepsRaw not None
            listSequenceTimeSteps = [sequenceTimeStepsRaw]
        else:
            listSequenceTimeSteps = []
        # get technologies of carrier
        technologiesCarrier = EnergySystem.get_technology_of_carrier(element.name)
        # if any technologies of carriers are aggregated 
        if technologiesCarrier:
            # iterate through technologies and extend listSequenceTimeSteps
            for technology in technologiesCarrier:
                listSequenceTimeSteps.append(Element.getAttributeOfSpecificElement(technology, "sequence_time_steps"))
            # get combined time steps, duration and sequence
            uniqueTimeStepSequences = TimeSeriesAggregation.uniqueTimeStepsInMultigrid(listSequenceTimeSteps)
            # if time steps of energy balance differ from carrier flows
            if uniqueTimeStepSequences:
                setTimeStepsOperation, timeStepsOperationDuration, sequence_time_steps = uniqueTimeStepSequences
                # set attributes
                TimeSeriesAggregation.setTimeAttributes(element,setTimeStepsOperation,timeStepsOperationDuration,sequence_time_steps,isEnergyBalance=True)

                # if carrier previously not aggregated
                if not element.isAggregated():
                    TimeSeriesAggregation.setTimeAttributes(element,setTimeStepsOperation,timeStepsOperationDuration,sequence_time_steps)
                # set aggregation indicator
                TimeSeriesAggregation.setAggregationIndicators(element, setEnergyBalanceIndicator=True)
                # iterate through raw time series data
#                 for timeSeries in element.raw_time_series:
#                     # if not yet aggregated, conduct "manual" time series aggregation
#                     if not element.isAggregated():
#                         dfTimeSeries = element.raw_time_series[timeSeries].loc[(slice(None), setTimeStepsOperation)]
#                     else:
#                         dfTimeSeries = getattr(element, timeSeries)
#                     # multiply with yearly variation
#                     dfTimeSeries = cls.multiplyYearlyVariation(element, timeSeries, dfTimeSeries)
#                     # save attribute
#                     setattr(element, timeSeries, dfTimeSeries)
            # no aggregation -> time steps and sequence of energy balance are equal to the carrier time steps and sequence
            else:
                # get time attributes of carrier
                setTimeStepsOperation, timeStepsOperationDuration, sequence_time_steps = TimeSeriesAggregation.getTimeAttributes(element)
                # set time attributes for energy balance
                TimeSeriesAggregation.setTimeAttributes(element, setTimeStepsOperation, timeStepsOperationDuration,sequence_time_steps, isEnergyBalance=True)
                # set aggregation indicator
                TimeSeriesAggregation.setAggregationIndicators(element, setEnergyBalanceIndicator=True)

    @classmethod
    def linkTimeSteps(cls, element):
        """ calculates the necessary overlapping time steps of the investment and operation of a technology for all years.
        It sets the union of the time steps for investment, operation and years.
        :param element: technology of the optimization """
        listSequenceTimeSteps = [
            EnergySystem.get_sequence_time_steps(element.name, "operation"),
            EnergySystem.get_sequence_time_steps(None, "yearly")
        ]

        uniqueTimeStepSequences = TimeSeriesAggregation.uniqueTimeStepsInMultigrid(listSequenceTimeSteps)
        if uniqueTimeStepSequences:
            set_time_steps, timeStepsDuration, sequence_time_steps = uniqueTimeStepSequences
            # set sequence time steps
            EnergySystem.set_sequence_time_steps(element.name, sequence_time_steps)
            # time series parameters
            cls.overwriteTimeSeriesWithExpandedTimeIndex(element, set_time_steps, sequence_time_steps)
            # set attributes
            TimeSeriesAggregation.setTimeAttributes(element,set_time_steps,timeStepsDuration,sequence_time_steps)
        else:
            # check to multiply the time series with the yearly variation
            cls.yearlyVariationNonaggregatedTimeSeries(element)

    @classmethod
    def calculateTimeStepsOperation2Invest(cls,element):
        """ calculates the conversion of operational time steps to invest time steps """
        _setTimeStepsOperation      = getattr(element,"setTimeStepsOperation")
        _sequenceTimeStepsOperation = getattr(element,"sequence_time_steps")
        _sequenceTimeStepsYearly    = getattr(EnergySystem.get_energy_system(),"sequence_time_steps_yearly")
        time_steps_operation2invest  = np.unique(np.vstack([_sequenceTimeStepsOperation,_sequenceTimeStepsYearly]),axis=1)
        time_steps_operation2invest  = {key:val for key,val in zip(time_steps_operation2invest[0,:],time_steps_operation2invest[1,:])}
        EnergySystem.set_time_steps_operation2invest(element.name,time_steps_operation2invest)

    @classmethod
    def overwriteTimeSeriesWithExpandedTimeIndex(cls, element, setTimeStepsOperation, sequence_time_steps):
        """ this method expands the aggregated time series to match the extended operational time steps because of matching the investment and operational time sequences.
        :param element: element of the optimization
        :param setTimeStepsOperation: new time steps operation
        :param sequence_time_steps: new order of operational time steps """
        headerSetTimeSteps      = EnergySystem.get_analysis()['headerDataInputs']["set_time_steps"]
        oldSequenceTimeSteps    = element.sequence_time_steps
        _idxOld2New             = np.array([np.unique(oldSequenceTimeSteps[np.argwhere(idx == sequence_time_steps)]) for idx in setTimeStepsOperation]).squeeze()
        for timeSeries in element.raw_time_series:
            _oldTimeSeries                  = getattr(element, timeSeries).unstack(headerSetTimeSteps)
            _newTimeSeries                  = pd.DataFrame(index=_oldTimeSeries.index, columns=setTimeStepsOperation)
            _newTimeSeries                  = _oldTimeSeries.loc[:,_idxOld2New[_newTimeSeries.columns]].T.reset_index(drop=True).T
            _newTimeSeries.columns.names    = [headerSetTimeSteps]
            _newTimeSeries                  = _newTimeSeries.stack()
            # multiply with yearly variation
            _newTimeSeries                  = cls.multiplyYearlyVariation(element, timeSeries, _newTimeSeries)
            # overwrite time series
            setattr(element, timeSeries, _newTimeSeries)

    @classmethod
    def yearlyVariationNonaggregatedTimeSeries(cls, element):
        """ multiply the time series with the yearly variation if the element's time series are not aggregated
        :param element: element of the optimization """
        for timeSeries in element.raw_time_series:
            # multiply with yearly variation
            _newTimeSeries = cls.multiplyYearlyVariation(element, timeSeries, getattr(element, timeSeries))
            # overwrite time series
            setattr(element, timeSeries, _newTimeSeries)

    @classmethod
    def multiplyYearlyVariation(cls, element, timeSeriesName, timeSeries):
        """ this method multiplies time series with the yearly variation of the time series
        The index of the variation is the same as the original time series, just time and year substituted
        :param element: technology of the optimization
        :param timeSeriesName: name of time series
        :param timeSeries: time series
        :return multipliedTimeSeries: timeSeries multiplied with yearly variation """
        if hasattr(element.datainput, timeSeriesName + "YearlyVariation"):
            _yearlyVariation            = getattr(element.datainput, timeSeriesName + "YearlyVariation")
            headerSetTimeSteps          = EnergySystem.get_analysis()['headerDataInputs']["set_time_steps"]
            headerSetTimeStepsYearly    = EnergySystem.get_analysis()['headerDataInputs']["set_time_steps_yearly"]
            _timeSeries                 = timeSeries.unstack(headerSetTimeSteps)
            _yearlyVariation            = _yearlyVariation.unstack(headerSetTimeStepsYearly)
            # if only one unique value
            if len(np.unique(_yearlyVariation)) == 1:
                timeSeries = _timeSeries.stack()*np.unique(_yearlyVariation)[0]
            else:
                for year in EnergySystem.get_energy_system().set_time_steps_yearly:
                    if not all(_yearlyVariation[year] == 1):
                        _base_time_steps                          = EnergySystem.decode_time_step(None, year, "yearly")
                        _elementTimeSteps                       = EnergySystem.encode_time_step(element.name, _base_time_steps, yearly=True)
                        _timeSeries.loc[:,_elementTimeSteps]    = _timeSeries[_elementTimeSteps].multiply(_yearlyVariation[year],axis=0).fillna(0)
                timeSeries = _timeSeries.stack()
        # round down if lower than decimal points
        _rounding_value                                  = 10 ** (-EnergySystem.get_solver()["roundingDecimalPointsTS"])
        timeSeries[timeSeries.abs() < _rounding_value]   = 0
        return timeSeries

    @classmethod
    def repeatSequenceTimeStepsForAllYears(cls):
        """ this method repeats the operational time series for all years."""
        logging.info("Repeat the time series sequences for all years")
        optimizedYears = len(EnergySystem.get_energy_system().set_time_steps_yearly)
        # concatenate the order of time steps for all elements and link with investment and yearly time steps
        for element in Element.get_all_elements():
            # optimizedYears              = EnergySystem.get_system()["optimizedYears"]
            oldSequenceTimeSteps        = Element.getAttributeOfSpecificElement(element.name, "sequence_time_steps")
            newSequenceTimeSteps        = np.hstack([oldSequenceTimeSteps] * optimizedYears)
            element.sequence_time_steps   = newSequenceTimeSteps
            EnergySystem.set_sequence_time_steps(element.name, element.sequence_time_steps)
            # calculate the time steps in operation to link with investment and yearly time steps
            cls.linkTimeSteps(element)
            # set operation2invest time step dict
            if element in Technology.get_all_elements():
                cls.calculateTimeStepsOperation2Invest(element)

    @classmethod
    def setAggregationIndicators(cls, element, setEnergyBalanceIndicator=False):
        """ this method sets the indicators that element is aggregated """
        # add order of time steps to Energy System
        EnergySystem.set_sequence_time_steps(element.name, element.sequence_time_steps, time_step_type="operation")
        # if energy balance indicator is set as well, save sequence of time steps in energy balance as well
        if setEnergyBalanceIndicator:
            EnergySystem.set_sequence_time_steps(element.name + "EnergyBalance", element.sequenceTimeStepsEnergyBalance,
                                              time_step_type="operation")
        element.setAggregated()

    @classmethod
    def uniqueTimeStepsInMultigrid(cls, listSequenceTimeSteps):
        """ this method returns the unique time steps of multiple time grids """
        sequence_time_steps           = np.zeros(np.size(listSequenceTimeSteps, axis=1)).astype(int)
        combinedSequenceTimeSteps   = np.vstack(listSequenceTimeSteps)
        uniqueCombinedTimeSteps, uniqueIndices, countCombinedTimeSteps = np.unique(combinedSequenceTimeSteps, axis=1,return_counts=True,return_index = True)
        # if unique time steps are the same as original, or if the second until last only have a single unique value
        # if combinedSequenceTimeSteps.shape == uniqueCombinedTimeSteps.shape:
        if len(np.unique(combinedSequenceTimeSteps[0,:])) == len(combinedSequenceTimeSteps[0,:]) or len(np.unique(combinedSequenceTimeSteps[1:,:],axis=1)[0]) == 1:
            return None
        set_time_steps      = []
        timeStepsDuration = {}
        for idxUniqueTimeStep, countUniqueTimeStep in enumerate(countCombinedTimeSteps):
            set_time_steps.append(idxUniqueTimeStep)
            timeStepsDuration[idxUniqueTimeStep]    = countUniqueTimeStep
            uniqueTimeStep                          = uniqueCombinedTimeSteps[:, idxUniqueTimeStep]
            idxInInput                              = np.argwhere(np.all(combinedSequenceTimeSteps.T == uniqueTimeStep, axis=1))
            # fill new order time steps 
            sequence_time_steps[idxInInput]           = idxUniqueTimeStep
        return (set_time_steps, timeStepsDuration, sequence_time_steps)

    @classmethod
    def overwriteRawTimeSeries(cls, element):
        """ this method overwrites the raw time series to the already once aggregated time series
        :param element: technology of the optimization """
        for timeSeries in element.raw_time_series:
            element.raw_time_series[timeSeries] = getattr(element, timeSeries)

    @staticmethod
    def setTimeAttributes(element,set_time_steps,timeStepsDuration,sequence_time_steps,isEnergyBalance = False):
        """ this method sets the operational time attributes of an element.
        :param element: element of the optimization
        :param set_time_steps: set_time_steps of operation
        :param timeStepsDuration: timeStepsDuration of operation
        :param sequence_time_steps: sequence of operation
        :param isEnergyBalance: boolean if attributes set for energyBalance """
        if isinstance(element,TimeSeriesAggregation):
            element.set_time_steps                    = set_time_steps
            element.timeStepsDuration               = timeStepsDuration
            element.sequence_time_steps               = sequence_time_steps
        elif not isEnergyBalance:
            element.setTimeStepsOperation           = set_time_steps
            element.timeStepsOperationDuration      = timeStepsDuration
            element.sequence_time_steps               = sequence_time_steps
        else:
            element.setTimeStepsEnergyBalance       = set_time_steps
            element.timeStepsEnergyBalanceDuration  = timeStepsDuration
            element.sequenceTimeStepsEnergyBalance  = sequence_time_steps

    @classmethod
    def setTimeSeriesAggregation(cls, timeSeriesAggregation):
        """ sets empty timeSeriesAggregation to timeSeriesAggregation
        :param timeSeriesAggregation: timeSeriesAggregation """
        cls.timeSeriesAggregation = timeSeriesAggregation

    @staticmethod
    def getTimeAttributes(element, isEnergyBalance=False):
        """ this method returns the operational time attributes of an element.
        :param element: element of the optimization
        :param isEnergyBalance: boolean if attributes set for energyBalance """
        if isinstance(element, TimeSeriesAggregation):
            return (element.set_time_steps,element.timeStepsDuration,element.sequence_time_steps)
        elif not isEnergyBalance:
            return (element.setTimeStepsOperation, element.timeStepsOperationDuration, element.sequence_time_steps)
        else:
            return (element.setTimeStepsEnergyBalance, element.timeStepsEnergyBalanceDuration, element.sequenceTimeStepsEnergyBalance)

    @classmethod
    def getTimeSeriesAggregation(cls):
        """ get timeSeriesAggregation
        :return timeSeriesAggregation: return timeSeriesAggregation """
        return cls.timeSeriesAggregation

    @classmethod
    def conduct_time_series_aggregation(cls):
        """ this method conducts time series aggregation """
        logging.info("\n--- Time series aggregation ---")
        timeSeriesAggregation = TimeSeriesAggregation()
        # repeat order of operational time steps and link with investment and yearly time steps
        cls.repeatSequenceTimeStepsForAllYears()
        logging.info("Calculate operational time steps for storage levels and energy balances")
        for element in StorageTechnology.get_all_elements():
            # calculate time steps of storage levels
            element.calculateTimeStepsStorageLevel(conductedTimeSeriesAggregation = timeSeriesAggregation.conductedTimeSeriesAggregation)
        # calculate new time steps of energy balance
        # for element in Carrier.get_all_elements():
            # cls.calculateTimeStepsEnergyBalance(element)

