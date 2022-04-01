"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to apply time series aggregation to time series
==========================================================================================================================================================================="""
import numpy as np
import pandas as pd
import logging
import tsam.timeseriesaggregation as tsam
from model.objects.energy_system import EnergySystem
from model.objects.element import Element
from model.objects.carrier import Carrier
from model.objects.technology.technology import Technology
from model.objects.technology.storage_technology import StorageTechnology

class TimeSeriesAggregation():
    def __init__(self):
        """ initializes the time series aggregation. The data is aggregated for a single year and then concatenated"""
        self.system                     = EnergySystem.getSystem()
        self.analysis                   = EnergySystem.getAnalysis()
        self.energySystem               = EnergySystem.getEnergySystem()
        self.headerSetTimeSteps         = self.analysis['headerDataInputs']["setTimeSteps"][0]
        # if setTimeSteps as input (because already aggregated), use this as base time step, otherwise self.setBaseTimeStepsYear
        self.setBaseTimeStepsYear       = list(range(0,self.system["timeStepsPerYear"]))
        self.numberTypicalPeriods       = min(self.system["timeStepsPerYear"], self.system["numberTimeStepsPerYearDefault"])
        self.numberTimeStepsPerPeriod   = 1
        # if number of time steps >= number of base time steps, skip aggregation
        if self.numberTypicalPeriods * self.numberTimeStepsPerPeriod < np.size(self.setBaseTimeStepsYear):
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
            self.setTimeSteps       = self.setBaseTimeStepsYear
            self.timeStepsDuration  = EnergySystem.calculateTimeStepDuration(self.setTimeSteps, self.setBaseTimeStepsYear)
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
            timeSeries          =self.dfTimeSeriesRaw,
            noTypicalPeriods    =self.numberTypicalPeriods,
            hoursPerPeriod      =self.numberTimeStepsPerPeriod,
            resolution          =self.analysis["timeSeriesAggregation"]["resolution"],
            clusterMethod       =self.analysis["timeSeriesAggregation"]["clusterMethod"],
            solver              =self.analysis["timeSeriesAggregation"]["solver"],
            extremePeriodMethod =self.analysis["timeSeriesAggregation"]["extremePeriodMethod"],
        )
        # create typical periods
        self.typicalPeriods     = self.aggregation.createTypicalPeriods().reset_index(drop=True)
        self.setTimeSteps       = self.aggregation.clusterPeriodIdx
        self.timeStepsDuration  = self.aggregation.clusterPeriodNoOccur
        self.orderTimeSteps     = self.aggregation.clusterOrder

    def setAggregatedTimeSeriesOfAllElements(self):
        """ this method sets the aggregated time series and sets the necessary attributes after the aggregation to a single time grid """
        for element in Element.getAllElements():
            rawTimeSeries = getattr(element, "rawTimeSeries")
            # setTimeSteps and duration
            if element.name in self.system["setCarriers"]:
                element.setTimeStepsCarrier         = list(self.setTimeSteps)
                element.timeStepsCarrierDuration    = self.timeStepsDuration
                element.orderTimeSteps              = self.orderTimeSteps
            elif element.name in self.system["setTechnologies"]:
                element.setTimeStepsOperation       = list(self.setTimeSteps)
                element.timeStepsOperationDuration  = self.timeStepsDuration
                element.orderTimeSteps              = self.orderTimeSteps
            else:
                raise KeyError(f"{element.name} neither in setCarriers nor setTechnologies")
            # iterate through raw time series
            for timeSeries in rawTimeSeries:
                _indexNames = list(rawTimeSeries[timeSeries].index.names)
                _indexNames.remove(self.headerSetTimeSteps)
                dfTimeSeries = rawTimeSeries[timeSeries].unstack(level=_indexNames)

                dfAggregatedTimeSeries      = pd.DataFrame(index=self.setTimeSteps, columns=dfTimeSeries.columns)
                # columns which are in aggregated time series and which are not
                if element.name in self.typicalPeriods and timeSeries in self.typicalPeriods[element.name]:
                    dfTypicalPeriods        = self.typicalPeriods[element.name, timeSeries]
                    AggregatedColumns       = dfTimeSeries.columns.intersection(dfTypicalPeriods.columns)
                    NotAggregatedColumns    = dfTimeSeries.columns.difference(dfTypicalPeriods.columns)
                    # aggregated columns
                    dfAggregatedTimeSeries[AggregatedColumns] = self.typicalPeriods[element.name, timeSeries][
                        AggregatedColumns]
                else:
                    NotAggregatedColumns = dfTimeSeries.columns
                # not aggregated columns
                dfAggregatedTimeSeries[NotAggregatedColumns] = dfTimeSeries.loc[dfAggregatedTimeSeries.index, NotAggregatedColumns]
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
            element.setTimeStepsEnergyBalance       = setTimeStepsCarrier
            element.timeStepsEnergyBalanceDuration  = timeStepsCarrierDuration
            element.orderTimeStepsEnergyBalance     = orderTimeStepsCarrier
            # if carrier previously not aggregated
            if not element.isAggregated():
                # iterate through raw time series data and conduct "manual" time series aggregation
                for timeSeries in element.rawTimeSeries:
                    dfTimeSeries = element.rawTimeSeries[timeSeries].loc[(slice(None), setTimeStepsCarrier)]
                    # save attribute
                    setattr(element, timeSeries, dfTimeSeries)
                element.setTimeStepsCarrier         = setTimeStepsCarrier
                element.timeStepsCarrierDuration    = timeStepsCarrierDuration
                element.orderTimeSteps              = orderTimeStepsCarrier
            # set aggregation indicator
            TimeSeriesAggregation.setAggregationIndicators(element, setEnergyBalanceIndicator=True)

    @classmethod
    def linkTimeSteps(cls, element):
        """ calculates the necessary overlapping time steps of the investment and operation of a technology for all year.
        It sets the union of the time steps for investment, operation and years.
        :param element: technology of the optimization """
        if element in Technology.getAllElements():
            listOrderTimeSteps = [
                EnergySystem.getOrderTimeSteps(element.name,"invest"),
                EnergySystem.getOrderTimeSteps(element.name,"operation"),
                EnergySystem.getOrderTimeSteps(None, "yearly")
            ]
        elif element in Carrier.getAllElements():
            listOrderTimeSteps = [
                EnergySystem.getOrderTimeSteps(element.name, "operation"),
                EnergySystem.getOrderTimeSteps(None, "yearly")
            ]
        setTimeSteps, timeStepsDuration, orderTimeSteps = \
            TimeSeriesAggregation.uniqueTimeStepsInMultigrid(listOrderTimeSteps)
        # time series parameters
        cls.overwriteTimeSeriesWithExpandedTimeIndex(element,setTimeSteps,orderTimeSteps)
        # set attributes
        if element in Technology.getAllElements():
            element.setTimeStepsOperation       = setTimeSteps
            element.timeStepsOperationDuration  = timeStepsDuration
            element.orderTimeSteps              = orderTimeSteps
        elif element in Carrier.getAllElements():
            element.setTimeStepsCarrier         = setTimeSteps
            element.timeStepsCarrierDuration    = timeStepsDuration
            element.orderTimeSteps              = orderTimeSteps
        EnergySystem.setOrderTimeSteps(element.name, element.orderTimeSteps)

    @classmethod
    def overwriteTimeSeriesWithExpandedTimeIndex(cls, element, setTimeStepsOperation, orderTimeSteps):
        """ this method expands the aggregated time series to match the extended operational time steps because of matching the investment and operational time sequences.
        :param element: technology of the optimization
        :param setTimeStepsOperation: new time steps operation
        :param orderTimeSteps: new order of operational time steps """
        headerSetTimeSteps = EnergySystem.getAnalysis()['headerDataInputs']["setTimeSteps"][0]
        oldOrderTimeSteps = element.orderTimeSteps
        for timeSeries in element.rawTimeSeries:
            _oldTimeSeries = getattr(element, timeSeries).unstack(headerSetTimeSteps)
            _newTimeSeries = pd.DataFrame(index=_oldTimeSeries.index, columns=setTimeStepsOperation)
            _idxOld2New = [np.unique(oldOrderTimeSteps[np.argwhere(idx == orderTimeSteps)]) for idx in
                           setTimeStepsOperation]
            _newTimeSeries = _newTimeSeries.apply(lambda row: _oldTimeSeries[_idxOld2New[row.name][0]], axis=0).stack()
            # overwrite time series
            setattr(element, timeSeries, _newTimeSeries)

    @classmethod
    def repeatOrderTimeStepsForAllYears(cls):
        """ this method repeats the operational time series for all years."""
        # concatenate the order of time steps for all elements and link with investment and yearly time steps
        for element in Element.getAllElements():
            timeStepsYearly         = EnergySystem.getSystem()["timeStepsYearly"]
            oldOrderTimeSteps       = Element.getAttributeOfSpecificElement(element.name, "orderTimeSteps")
            newOrderTimeSteps       = np.hstack([oldOrderTimeSteps]*timeStepsYearly)
            element.orderTimeSteps  = newOrderTimeSteps
            EnergySystem.setOrderTimeSteps(element.name, element.orderTimeSteps)
            # calculate the time steps in operation to link with investment and yearly time steps
            cls.linkTimeSteps(element)

    @classmethod
    def setAggregationIndicators(cls, element, setEnergyBalanceIndicator=False):
        """ this method sets the indicators that element is aggregated """
        # add order of time steps to Energy System
        EnergySystem.setOrderTimeSteps(element.name, element.orderTimeSteps, timeStepType="operation")
        # if energy balance indicator is set as well, save order of time steps in energy balance as well
        if setEnergyBalanceIndicator:
            EnergySystem.setOrderTimeSteps(element.name + "EnergyBalance", element.orderTimeStepsEnergyBalance,
                                           timeStepType="operation")
        element.setAggregated()

    @classmethod
    def uniqueTimeStepsInMultigrid(cls, listOrderTimeSteps):
        """ this method returns the unique time steps of multiple time grids """
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
        """ this method overwrites the raw time series to the already once aggregated time series
        :param element: technology of the optimization """
        for timeSeries in element.rawTimeSeries:
            element.rawTimeSeries[timeSeries] = getattr(element, timeSeries)

    @classmethod
    def conductTimeSeriesAggregation(cls):
        """ this method conducts time series aggregation """
        logging.info("\n--- Time series aggregation ---")
        TimeSeriesAggregation()
        # repeat order of operational time steps and link with investment and yearly time steps
        cls.repeatOrderTimeStepsForAllYears()
        for element in StorageTechnology.getAllElements():
            # calculate time steps of storage levels
            element.calculateTimeStepsStorageLevel()
        # calculate new time steps of energy balance
        for element in Carrier.getAllElements():
            cls.calculateTimeStepsEnergyBalance(element)
