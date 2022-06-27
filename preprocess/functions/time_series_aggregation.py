"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Functions to apply time series aggregation to time series
==========================================================================================================================================================================="""
import pandas as pd
import numpy as np
import logging
import tsam.timeseriesaggregation as tsam

from model.objects.energy_system import EnergySystem
from model.objects.element import Element
from model.objects.carrier import Carrier
from model.objects.technology.technology import Technology
from model.objects.technology.storage_technology import StorageTechnology


class TimeSeriesAggregation():
    timeSeriesAggregation = None

    def __init__(self):
        """ initializes the time series aggregation. The data is aggregated for a single year and then concatenated"""
        self.system                 = EnergySystem.getSystem()
        self.analysis               = EnergySystem.getAnalysis()
        self.energySystem           = EnergySystem.getEnergySystem()
        self.headerSetTimeSteps     = self.analysis['headerDataInputs']["setTimeSteps"][0]
        # if setTimeSteps as input (because already aggregated), use this as base time step, otherwise self.setBaseTimeSteps
        self.setBaseTimeSteps       = self.energySystem.setBaseTimeStepsYearly
        self.numberTypicalPeriods   = min(self.system["unaggregatedTimeStepsPerYear"], self.system["aggregatedTimeStepsPerYear"])
        # set timeSeriesAggregation
        TimeSeriesAggregation.setTimeSeriesAggregation(self)
        # if number of time steps >= number of base time steps, skip aggregation
        if self.numberTypicalPeriods < np.size(self.setBaseTimeSteps) and self.system["conductTimeSeriesAggregation"]:
            # select time series
            self.selectTimeSeriesOfAllElements()
            if not self.dfTimeSeriesRaw.empty:
                # run time series aggregation to create typical periods
                self.runTimeSeriesAggregation()
        else:
            self.typicalPeriods = pd.DataFrame()
            _setTimeSteps       = self.setBaseTimeSteps
            _timeStepDuration   = EnergySystem.calculateTimeStepDuration(_setTimeSteps,self.setBaseTimeSteps)
            _sequenceTimeSteps  = np.concatenate([[timeStep] * _timeStepDuration[timeStep] for timeStep in _timeStepDuration])
            TimeSeriesAggregation.setTimeAttributes(self,_setTimeSteps,_timeStepDuration,_sequenceTimeSteps)
            # set aggregated time series
            self.setAggregatedTimeSeriesOfAllElements()

    def selectTimeSeriesOfAllElements(self):
        """ this method retrieves the raw time series for the aggregation of all input data sets.
        Only in aligned time grid approach! """

        _dictRawTimeSeries = {}
        for element in Element.getAllElements():
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

    def setAggregatedTimeSeriesOfAllElements(self):
        """ this method sets the aggregated time series and sets the necessary attributes after the aggregation to a single time grid """
        for element in Element.getAllElements():
            rawTimeSeries = getattr(element, "rawTimeSeries")
            # setTimeSteps, duration and sequence time steps
            element.setTimeStepsOperation       = list(self.setTimeSteps)
            element.timeStepsOperationDuration  = self.timeStepsDuration
            element.sequenceTimeSteps           = self.sequenceTimeSteps

            # iterate through raw time series
            for timeSeries in rawTimeSeries:
                _indexNames = list(rawTimeSeries[timeSeries].index.names)
                _indexNames.remove(self.headerSetTimeSteps)
                dfTimeSeries = rawTimeSeries[timeSeries].unstack(level=_indexNames)

                dfAggregatedTimeSeries = pd.DataFrame(index=self.setTimeSteps, columns=dfTimeSeries.columns)
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
                dfAggregatedTimeSeries.columns.names    = _indexNames
                dfAggregatedTimeSeries                  = dfAggregatedTimeSeries.stack(_indexNames)
                dfAggregatedTimeSeries.index            = dfAggregatedTimeSeries.index.reorder_levels(
                                                            _indexNames + [self.headerSetTimeSteps])
                setattr(element, timeSeries, dfAggregatedTimeSeries)
                TimeSeriesAggregation.setAggregationIndicators(element)

    @classmethod
    def extractRawTimeSeries(cls, element, headerSetTimeSteps):
        """ extract the time series from an element and concatenates the non-constant time series to a pd.DataFrame
        :param element: element of the optimization 
        :param headerSetTimeSteps: name of setTimeSteps
        :return dfTimeSeriesRaw: pd.DataFrame with non-constant time series"""
        _dictRawTimeSeries = {}
        rawTimeSeries = getattr(element, "rawTimeSeries")
        for timeSeries in rawTimeSeries:
            rawTimeSeries[timeSeries].name = timeSeries
            _indexNames = list(rawTimeSeries[timeSeries].index.names)
            _indexNames.remove(headerSetTimeSteps)
            dfTimeSeries = rawTimeSeries[timeSeries].unstack(level=_indexNames)
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
            sequenceTimeStepsRaw = element.sequenceTimeSteps
            # if sequenceTimeStepsRaw not None
            listSequenceTimeSteps = [sequenceTimeStepsRaw]
        else:
            listSequenceTimeSteps = []
        # get technologies of carrier
        technologiesCarrier = EnergySystem.getTechnologyOfCarrier(element.name)
        # if any technologies of carriers are aggregated 
        if technologiesCarrier:
            # iterate through technologies and extend listSequenceTimeSteps
            for technology in technologiesCarrier:
                listSequenceTimeSteps.append(Element.getAttributeOfSpecificElement(technology, "sequenceTimeSteps"))
            # get combined time steps, duration and sequence
            setTimeStepsOperation, timeStepsOperationDuration, sequenceTimeSteps = TimeSeriesAggregation.uniqueTimeStepsInMultigrid(
                listSequenceTimeSteps)
            # set attributes
            TimeSeriesAggregation.setTimeAttributes(element,setTimeStepsOperation,timeStepsOperationDuration,sequenceTimeSteps,isEnergyBalance=True)

            # if carrier previously not aggregated
            if not element.isAggregated():
                TimeSeriesAggregation.setTimeAttributes(element,setTimeStepsOperation,timeStepsOperationDuration,sequenceTimeSteps)

            # iterate through raw time series data
            for timeSeries in element.rawTimeSeries:
                # if not yet aggregated, conduct "manual" time series aggregation
                if not element.isAggregated():
                    dfTimeSeries = element.rawTimeSeries[timeSeries].loc[(slice(None), setTimeStepsOperation)]
                else:
                    dfTimeSeries = getattr(element, timeSeries)
                # save attribute
                setattr(element, timeSeries, dfTimeSeries)
            # set aggregation indicator
            TimeSeriesAggregation.setAggregationIndicators(element, setEnergyBalanceIndicator=True)
            # multiply with yearly variation
            for timeSeries in element.rawTimeSeries:
                dfTimeSeries = cls.multiplyYearlyVariation(element, timeSeries, dfTimeSeries)

    @classmethod
    def linkTimeSteps(cls, element):
        """ calculates the necessary overlapping time steps of the investment and operation of a technology for all year.
        It sets the union of the time steps for investment, operation and years.
        :param element: technology of the optimization """
        listSequenceTimeSteps = [
            EnergySystem.getSequenceTimeSteps(element.name, "operation"),
            EnergySystem.getSequenceTimeSteps(None, "yearly")
        ]
        if element in Technology.getAllElements():
            listSequenceTimeSteps.append(EnergySystem.getSequenceTimeSteps(element.name, "invest"))

        setTimeSteps, timeStepsDuration, sequenceTimeSteps = \
            TimeSeriesAggregation.uniqueTimeStepsInMultigrid(listSequenceTimeSteps)
        # set sequence time steps
        EnergySystem.setSequenceTimeSteps(element.name, sequenceTimeSteps)
        # time series parameters

        cls.overwriteTimeSeriesWithExpandedTimeIndex(element, setTimeSteps, sequenceTimeSteps)

        # set attributes
        TimeSeriesAggregation.setTimeAttributes(element,setTimeSteps,timeStepsDuration,sequenceTimeSteps)

    @classmethod
    def calculateTimeStepsOperation2Invest(cls,element):
        """ calculates the conversion of operational time steps to invest time steps """
        _setTimeStepsOperation      = getattr(element,"setTimeStepsOperation")
        _sequenceTimeStepsOperation = getattr(element,"sequenceTimeSteps")
        _sequenceTimeStepsInvest    = getattr(element, "sequenceTimeStepsInvest")
        _timeStepsOperation2Invest  = {}
        for timeStep in _setTimeStepsOperation:
            convertedTimeSteps = np.unique(_sequenceTimeStepsInvest[_sequenceTimeStepsOperation == timeStep])
            assert len(convertedTimeSteps) == 1, f"more than one invest time step per operational time step. Impossible, here to debug"
            _timeStepsOperation2Invest[timeStep] = convertedTimeSteps[0]
        EnergySystem.setTimeStepsOperation2Invest(element.name,_timeStepsOperation2Invest)

    @classmethod
    def overwriteTimeSeriesWithExpandedTimeIndex(cls, element, setTimeStepsOperation, sequenceTimeSteps):
        """ this method expands the aggregated time series to match the extended operational time steps because of matching the investment and operational time sequences.
        :param element: technology of the optimization
        :param setTimeStepsOperation: new time steps operation
        :param sequenceTimeSteps: new order of operational time steps """
        headerSetTimeSteps = EnergySystem.getAnalysis()['headerDataInputs']["setTimeSteps"][0]
        oldSequenceTimeSteps = element.sequenceTimeSteps
        for timeSeries in element.rawTimeSeries:
            _oldTimeSeries = getattr(element, timeSeries).unstack(headerSetTimeSteps)
            _newTimeSeries = pd.DataFrame(index=_oldTimeSeries.index, columns=setTimeStepsOperation)
            _idxOld2New = np.array([np.unique(oldSequenceTimeSteps[np.argwhere(idx == sequenceTimeSteps)]) for idx in setTimeStepsOperation]).squeeze()
            _newTimeSeries = _oldTimeSeries.loc[:,_idxOld2New[_newTimeSeries.columns]].T.reset_index(drop=True).T
            _newTimeSeries.columns.names = [headerSetTimeSteps]
            _newTimeSeries = _newTimeSeries.stack()
            # multiply with yearly variation
            _newTimeSeries = cls.multiplyYearlyVariation(element, timeSeries, _newTimeSeries)
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
        if hasattr(element.dataInput, timeSeriesName + "YearlyVariation"):
            _yearlyVariation = getattr(element.dataInput, timeSeriesName + "YearlyVariation")
            headerSetTimeSteps = EnergySystem.getAnalysis()['headerDataInputs']["setTimeSteps"][0]
            headerSetTimeStepsYearly = EnergySystem.getAnalysis()['headerDataInputs']["setTimeStepsYearly"][0]
            _timeSeries = timeSeries.unstack(headerSetTimeSteps)
            _yearlyVariation = _yearlyVariation.unstack(headerSetTimeStepsYearly)
            # if only one unique value
            if len(np.unique(_yearlyVariation)) == 1:
                return _timeSeries*np.unique(_yearlyVariation)[0]
            else:
                for year in EnergySystem.getEnergySystem().setTimeStepsYearly:
                    if not all(_yearlyVariation[year] == 1):
                        _baseTimeSteps = EnergySystem.decodeTimeStep(None, year, "yearly")
                        _elementTimeSteps = EnergySystem.encodeTimeStep(element.name, _baseTimeSteps, yearly=True)
                        # _npTimeSeries = _timeSeries[_elementTimeSteps].to_numpy()
                        # _npYearlyVariation = np.reshape(_yearlyVariation[year].to_numpy(),(-1,1))
                        _timeSeries.loc[:,_elementTimeSteps] = _timeSeries[_elementTimeSteps].multiply(_yearlyVariation[year],axis=0).fillna(0)
                timeSeries = _timeSeries.stack()
                return timeSeries
        else:
            return timeSeries

    @classmethod
    def repeatSequenceTimeStepsForAllYears(cls):
        """ this method repeats the operational time series for all years."""
        # concatenate the order of time steps for all elements and link with investment and yearly time steps
        for element in Element.getAllElements():
            optimizedYears = EnergySystem.getSystem()["optimizedYears"]
            oldSequenceTimeSteps = Element.getAttributeOfSpecificElement(element.name, "sequenceTimeSteps")
            newSequenceTimeSteps = np.hstack([oldSequenceTimeSteps] * optimizedYears)
            element.sequenceTimeSteps = newSequenceTimeSteps
            EnergySystem.setSequenceTimeSteps(element.name, element.sequenceTimeSteps)
            # calculate the time steps in operation to link with investment and yearly time steps
            cls.linkTimeSteps(element)
            # set operation2invest time step dict
            if element in Technology.getAllElements():
                cls.calculateTimeStepsOperation2Invest(element)

    @classmethod
    def setAggregationIndicators(cls, element, setEnergyBalanceIndicator=False):
        """ this method sets the indicators that element is aggregated """
        # add order of time steps to Energy System
        EnergySystem.setSequenceTimeSteps(element.name, element.sequenceTimeSteps, timeStepType="operation")
        # if energy balance indicator is set as well, save sequence of time steps in energy balance as well
        if setEnergyBalanceIndicator:
            EnergySystem.setSequenceTimeSteps(element.name + "EnergyBalance", element.sequenceTimeStepsEnergyBalance,
                                              timeStepType="operation")
        element.setAggregated()

    @classmethod
    def uniqueTimeStepsInMultigrid(cls, listSequenceTimeSteps):
        """ this method returns the unique time steps of multiple time grids """
        sequenceTimeSteps = np.zeros(np.size(listSequenceTimeSteps, axis=1)).astype(int)
        combinedSequenceTimeSteps = np.vstack(listSequenceTimeSteps)
        uniqueCombinedTimeSteps, countCombinedTimeSteps = np.unique(combinedSequenceTimeSteps, axis=1,
                                                                    return_counts=True)
        setTimeSteps      = []
        timeStepsDuration = {}
        for idxUniqueTimeStep, countUniqueTimeStep in enumerate(countCombinedTimeSteps):
            setTimeSteps.append(idxUniqueTimeStep)
            timeStepsDuration[idxUniqueTimeStep] = countUniqueTimeStep
            uniqueTimeStep = uniqueCombinedTimeSteps[:, idxUniqueTimeStep]
            idxInInput = np.argwhere(np.all(combinedSequenceTimeSteps.T == uniqueTimeStep, axis=1))
            # fill new order time steps 
            sequenceTimeSteps[idxInInput] = idxUniqueTimeStep
        return setTimeSteps, timeStepsDuration, sequenceTimeSteps

    @classmethod
    def overwriteRawTimeSeries(cls, element):
        """ this method overwrites the raw time series to the already once aggregated time series
        :param element: technology of the optimization """
        for timeSeries in element.rawTimeSeries:
            element.rawTimeSeries[timeSeries] = getattr(element, timeSeries)

    @staticmethod
    def setTimeAttributes(element,setTimeSteps,timeStepsDuration,sequenceTimeSteps,isEnergyBalance = False):
        """ this method sets the operational time attributes of an element.
        :param element: element of the optimization
        :param setTimeSteps: setTimeSteps of operation
        :param timeStepsDuration: timeStepsDuration of operation
        :param sequenceTimeSteps: sequence of operation
        :param isEnergyBalance: boolean if attributes set for energyBalance """
        if isinstance(element,TimeSeriesAggregation):
            element.setTimeSteps                    = setTimeSteps
            element.timeStepsDuration               = timeStepsDuration
            element.sequenceTimeSteps               = sequenceTimeSteps
        elif not isEnergyBalance:
            element.setTimeStepsOperation           = setTimeSteps
            element.timeStepsOperationDuration      = timeStepsDuration
            element.sequenceTimeSteps               = sequenceTimeSteps
        else:
            element.setTimeStepsEnergyBalance       = setTimeSteps
            element.timeStepsEnergyBalanceDuration  = timeStepsDuration
            element.sequenceTimeStepsEnergyBalance  = sequenceTimeSteps

    @classmethod
    def setTimeSeriesAggregation(cls, timeSeriesAggregation):
        """ sets empty timeSeriesAggregation to timeSeriesAggregation
        :param timeSeriesAggregation: timeSeriesAggregation """
        cls.timeSeriesAggregation = timeSeriesAggregation

    @classmethod
    def getTimeSeriesAggregation(cls):
        """ get timeSeriesAggregation
        :return timeSeriesAggregation: return timeSeriesAggregation """
        return cls.timeSeriesAggregation

    @classmethod
    def conductTimeSeriesAggregation(cls):
        """ this method conducts time series aggregation """
        logging.info("\n--- Time series aggregation ---")
        TimeSeriesAggregation()
        # repeat order of operational time steps and link with investment and yearly time steps
        cls.repeatSequenceTimeStepsForAllYears()
        for element in StorageTechnology.getAllElements():
            # calculate time steps of storage levels
            element.calculateTimeStepsStorageLevel()
        # calculate new time steps of energy balance
        for element in Carrier.getAllElements():
            cls.calculateTimeStepsEnergyBalance(element)

