"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to apply time series aggregation to time series
==========================================================================================================================================================================="""
import numpy as np
import pandas as pd
import copy
import logging
import tsam.timeseriesaggregation as tsam
from model.objects.carrier import Carrier

from model.objects.energy_system import EnergySystem
from model.objects.element import Element
from model.objects.technology.storage_technology import StorageTechnology


# noinspection PyAttributeOutsideInit
class TimeSeriesAggregation():
    def __init__(self):
        self.system = EnergySystem.getSystem()
        self.analysis = EnergySystem.getAnalysis()
        self.headerSetTimeSteps = self.analysis['headerDataInputs']["setTimeSteps"][0]
        # if setTimeSteps as input (because already aggregated), use this as base time step, otherwise self.setBaseTimeSteps
        self.setBaseTimeSteps = self.system["setTimeSteps"]
        self.numberTypicalPeriods = min(self.system["numberTimeStepsDefault"], np.size(self.setBaseTimeSteps))
        self.numberTimeStepsPerPeriod = 1
        # if number of time steps >= number of base time steps, skip aggregation
        if self.numberTypicalPeriods * self.numberTimeStepsPerPeriod < np.size(self.setBaseTimeSteps):
            # select time series
            self.selectTimeSeriesOfAllElements()
            if not self.dfTimeSeriesRaw.empty:
                # conduct time series aggregation
                # substitute column names
                self.substituteColumnNames(direction="flatten")
                # run time series aggregation to create typical periods
                self.runTimeSeriesAggregation()
                # resubstitute column names
                self.substituteColumnNames(direction="raise")
                # set aggregated time series
                self.setAggregatedTimeSeriesOfAllElements()
        else:
            self.typicalPeriods     = pd.DataFrame()
            self.setTimeSteps       = self.setBaseTimeSteps
            self.timeStepsDuration  = EnergySystem.calculateTimeStepDuration(self.setTimeSteps, self.setBaseTimeSteps)
            self.orderTimeSteps     = np.concatenate(
                [[timeStep] * self.timeStepsDuration[timeStep] for timeStep in self.timeStepsDuration])
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
            self.columnNamesOriginal = self.dfTimeSeriesRaw.columns
            self.columnNamesFlat = [str(index) for index in self.columnNamesOriginal]
            self.dfTimeSeriesRaw.columns = self.columnNamesFlat
        elif direction == "raise":
            self.typicalPeriods = self.typicalPeriods[self.columnNamesFlat]
            self.typicalPeriods.columns = self.columnNamesOriginal

    def runTimeSeriesAggregation(self):
        """ this method runs the time series aggregation """

        # create aggregation object
        self.aggregation = tsam.TimeSeriesAggregation(
            timeSeries=self.dfTimeSeriesRaw,
            noTypicalPeriods=self.numberTypicalPeriods,
            hoursPerPeriod=self.numberTimeStepsPerPeriod,
            resolution=self.analysis["timeSeriesAggregation"]["resolution"],
            clusterMethod=self.analysis["timeSeriesAggregation"]["clusterMethod"],
            solver=self.analysis["timeSeriesAggregation"]["solver"],
            extremePeriodMethod=self.analysis["timeSeriesAggregation"]["extremePeriodMethod"],
        )
        # create typical periods
        self.typicalPeriods = self.aggregation.createTypicalPeriods().reset_index(drop=True)
        self.setTimeSteps = self.aggregation.clusterPeriodIdx
        self.timeStepsDuration = self.aggregation.clusterPeriodNoOccur
        self.orderTimeSteps = self.aggregation.clusterOrder

    def setAggregatedTimeSeriesOfAllElements(self):
        """ this method sets the aggregated time series and sets the necessary attributes after the aggregation to a single time grid """
        for element in Element.getAllElements():
            rawTimeSeries = getattr(element, "rawTimeSeries")
            # setTimeSteps and duration
            if element.name in self.system["setCarriers"]:
                element.setTimeStepsCarrier = list(self.setTimeSteps)
                element.timeStepsCarrierDuration = self.timeStepsDuration
                element.orderTimeSteps = self.orderTimeSteps
            elif element.name in self.system["setTechnologies"]:
                element.setTimeStepsOperation = list(self.setTimeSteps)
                element.timeStepsOperationDuration = self.timeStepsDuration
                element.orderTimeSteps = self.orderTimeSteps
            else:
                raise KeyError(f"{element.name} neither in setCarriers nor setTechnologies")
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
                dfAggregatedTimeSeries[NotAggregatedColumns] = dfTimeSeries[NotAggregatedColumns]
                # reorder
                dfAggregatedTimeSeries.index.names = [self.headerSetTimeSteps]
                dfAggregatedTimeSeries.columns.names = _indexNames
                dfAggregatedTimeSeries = dfAggregatedTimeSeries.stack(_indexNames)
                dfAggregatedTimeSeries.index = dfAggregatedTimeSeries.index.reorder_levels(
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
            orderTimeStepsRaw = element.orderTimeSteps
            # if orderTimeStepsRaw not None
            listOrderTimeSteps = [orderTimeStepsRaw]
        else:
            listOrderTimeSteps = []
        # get technologies of carrier
        technologiesCarrier = EnergySystem.getTechnologyOfCarrier(element.name)
        # if any technologies of carriers are aggregated 
        if technologiesCarrier:
            # iterate through technologies and extend listOrderTimeSteps
            for technology in technologiesCarrier:
                listOrderTimeSteps.append(Element.getAttributeOfSpecificElement(technology, "orderTimeSteps"))
            # get combined time steps, duration and order
            setTimeStepsCarrier, timeStepsCarrierDuration, orderTimeStepsCarrier = TimeSeriesAggregation.uniqueTimeStepsInMultigrid(
                listOrderTimeSteps)
            # set attributes
            element.setTimeStepsEnergyBalance = setTimeStepsCarrier
            element.timeStepsEnergyBalanceDuration = timeStepsCarrierDuration
            element.orderTimeStepsEnergyBalance = orderTimeStepsCarrier
            # if carrier previously not aggregated
            if not element.isAggregated():
                # iterate through raw time series data and conduct "manual" time series aggregation
                for timeSeries in element.rawTimeSeries:
                    dfTimeSeries = element.rawTimeSeries[timeSeries].loc[(slice(None), setTimeStepsCarrier)]
                    # save attribute
                    setattr(element, timeSeries, dfTimeSeries)
                element.setTimeStepsCarrier = setTimeStepsCarrier
                element.timeStepsCarrierDuration = timeStepsCarrierDuration
                element.orderTimeSteps = orderTimeStepsCarrier
            # set aggregation indicator
            TimeSeriesAggregation.setAggregationIndicators(element, setEnergyBalanceIndicator=True)

    @classmethod
    def setAggregationIndicators(cls, element, setEnergyBalanceIndicator=False):
        """ this method sets the indicators that element is aggregated """
        system = EnergySystem.getSystem()
        # add order of time steps to Energy System
        EnergySystem.setOrderTimeSteps(element.name, element.orderTimeSteps, timeStepType="operation")
        # if energy balance indicator is set as well, save order of time steps in energy balance as well
        if setEnergyBalanceIndicator:
            EnergySystem.setOrderTimeSteps(element.name + "EnergyBalance", element.orderTimeStepsEnergyBalance,
                                           timeStepType="operation")
            # if technology, add to technologyOfCarrier list
        if element.name in system["setTechnologies"]:
            if element.name in system["setConversionTechnologies"]:
                EnergySystem.setTechnologyOfCarrier(element.name, element.inputCarrier + element.outputCarrier)
            else:
                EnergySystem.setTechnologyOfCarrier(element.name, element.referenceCarrier)
        # set the aggregation status of element to true
        element.setAggregated()

    @classmethod
    def uniqueTimeStepsInMultigrid(cls, listOrderTimeSteps):
        """ this method returns the unique time steps of multiple time grids """
        system = EnergySystem.getSystem()
        orderTimeSteps = np.zeros(np.size(listOrderTimeSteps, axis=1)).astype(int)
        combinedOrderTimeSteps = np.vstack(listOrderTimeSteps)
        uniqueCombinedTimeSteps, countCombinedTimeSteps = np.unique(combinedOrderTimeSteps, axis=1, return_counts=True)
        setTimeSteps = []
        timeStepsDuration = {}
        for idxUniqueTimeStep, countUniqueTimeStep in enumerate(countCombinedTimeSteps):
            setTimeSteps.append(idxUniqueTimeStep)
            timeStepsDuration[idxUniqueTimeStep] = countUniqueTimeStep
            uniqueTimeStep = uniqueCombinedTimeSteps[:, idxUniqueTimeStep]
            idxInInput = np.argwhere(np.all(combinedOrderTimeSteps.T == uniqueTimeStep, axis=1))
            # fill new order time steps 
            orderTimeSteps[idxInInput] = idxUniqueTimeStep
        return setTimeSteps, timeStepsDuration, orderTimeSteps

    @classmethod
    def overwriteRawTimeSeries(cls, element):
        """ this method overwrites the raw time series to the already once aggregated time series """
        for timeSeries in element.rawTimeSeries:
            element.rawTimeSeries[timeSeries] = getattr(element, timeSeries)

    @classmethod
    def conductTimeSeriesAggregation(cls):
        """ this method conducts time series aggregation """
        TimeSeriesAggregation()

        # calculate storage level time steps
        for tech in StorageTechnology.getAllElements():
            # calculate time steps of storage levels
            tech.calculateTimeStepsStorageLevel()
