"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to apply time series aggregation to time series
==========================================================================================================================================================================="""
import numpy as np
from scipy.stats import linregress
import pandas as pd
import os
import logging
import tsam.timeseriesaggregation as tsam

from model.objects.energy_system import EnergySystem

class TimeSeriesAggregation():
    def __init__(self,element,inputPath):
        self.element            = element
        self.inputPath          = inputPath
        self.dataInput          = element.dataInput
        self.setBaseTimeSteps   = EnergySystem.getEnergySystem().setBaseTimeSteps
        self.system             = EnergySystem.getSystem()
        self.analysis           = EnergySystem.getAnalysis()
        self.headerSetTimeSteps = self.analysis['headerDataInputs']["setTimeSteps"][0]
        # get number of time steps
        self.getNumberOfTimeSteps()
        # select time series
        self.selectTimeSeries()
        # if raw nonconstant time series exist
        if not self.dfTimeSeriesRaw.empty:
            # substitute column names
            self.substituteColumnNames(direction="flatten")
            # run time series aggregation to create typical periods
            self.runTimeSeriesAggregation()
            # resubstitute column names
            self.substituteColumnNames(direction="raise")
        # set aggregated time series
        self.setAggregatedTimeSeries()
        # add order of time steps to Energy System
        EnergySystem.setOrderTimeSteps(self.element.name,self.element.orderTimeSteps,timeStepType="operation") 

    def getNumberOfTimeSteps(self):
        """ this method extracts number of time steps for time series aggregation """
        if self.element.name in self.system["setCarriers"]: 
            typeOfTimeSteps = None
        elif self.element.name in self.system["setTechnologies"]:
            typeOfTimeSteps = "operation"
        else:
            raise KeyError(f"{self.element.name} neither in setCarriers nor setTechnologies")
        self.numberTypicalPeriods,self.numberTimeStepsPerPeriod = self.dataInput.extractNumberOfTimeSteps(self.inputPath,typeOfTimeSteps)
    
    def selectTimeSeries(self):
        """ this method selects the time series of the input element and creates a common dataframe"""
        dictRawtimeSeries   = {}
        rawTimeSeries       = getattr(self.element,"rawTimeSeries")
        for timeSeries in rawTimeSeries:
            rawTimeSeries[timeSeries].name  = timeSeries
            _indexNames                     = list(rawTimeSeries[timeSeries].index.names)
            _indexNames.remove(self.headerSetTimeSteps)
            dfTimeSeries                    = rawTimeSeries[timeSeries].unstack(level = _indexNames)
            # select time series that are not constant (rows have more than 1 unique entries)
            dfTimeSeriesNonConstant         = dfTimeSeries[dfTimeSeries.columns[dfTimeSeries.apply(lambda column: len(np.unique(column))!=1)]]
            dictRawtimeSeries[timeSeries]   = dfTimeSeriesNonConstant
        self.dfTimeSeriesRaw = pd.concat(dictRawtimeSeries.values(),axis=1,keys=dictRawtimeSeries.keys())

    def substituteColumnNames(self,direction = "flatten"):
        """ this method substitutes the column names to have flat columns names (otherwise sklearn warning) """
        if direction == "flatten":
            self.columnNamesOriginal        = self.dfTimeSeriesRaw.columns
            self.columnNamesFlat            = [str(index) for index in self.columnNamesOriginal]
            self.dfTimeSeriesRaw.columns    = self.columnNamesFlat
        elif direction == "raise":
            self.typicalPeriods = self.typicalPeriods[self.columnNamesFlat]
            self.typicalPeriods.columns = self.columnNamesOriginal
         

    def runTimeSeriesAggregation(self):
        """ this method runs the time series aggregation """
        # create aggregation object
        self.aggregation = tsam.TimeSeriesAggregation(
            timeSeries          = self.dfTimeSeriesRaw,
            noTypicalPeriods    = self.numberTypicalPeriods,
            hoursPerPeriod      = self.numberTimeStepsPerPeriod,
            resolution          = self.analysis["timeSeriesAggregation"]["resolution"],
            clusterMethod       = self.analysis["timeSeriesAggregation"]["clusterMethod"],
            solver              = self.analysis["timeSeriesAggregation"]["solver"],
            extremePeriodMethod = self.analysis["timeSeriesAggregation"]["extremePeriodMethod"],
        )
        # create typical periods
        self.typicalPeriods     = self.aggregation.createTypicalPeriods().reset_index(drop=True)
    
    def setAggregatedTimeSeries(self):
        """ this method sets the aggregated time series and sets the necessary attributes"""
        rawTimeSeries       = getattr(self.element,"rawTimeSeries")
        # if time series were aggregated
        if not self.dfTimeSeriesRaw.empty:
            # setTimeSteps and duration
            if self.element.name in self.system["setCarriers"]: 
                self.element.setTimeStepsCarrier        = list(self.aggregation.clusterPeriodIdx)
                self.element.timeStepsCarrierDuration   = self.aggregation.clusterPeriodNoOccur
                self.element.orderTimeSteps             = self.aggregation.clusterOrder
            elif self.element.name in self.system["setTechnologies"]:
                self.element.setTimeStepsOperation      = list(self.aggregation.clusterPeriodIdx)
                self.element.timeStepsOperationDuration = self.aggregation.clusterPeriodNoOccur
                self.element.orderTimeSteps             = self.aggregation.clusterOrder
            else:
                raise KeyError(f"{self.element.name} neither in setCarriers nor setTechnologies")
            # iterate through raw time series
            for timeSeries in rawTimeSeries:
                _indexNames = list(rawTimeSeries[timeSeries].index.names)
                _indexNames.remove(self.headerSetTimeSteps)
                dfTimeSeries = rawTimeSeries[timeSeries].unstack(level = _indexNames)
                
                dfAggregatedTimeSeries = pd.DataFrame(index=self.aggregation.clusterPeriodIdx,columns=dfTimeSeries.columns)
                # columns which are in aggregated time series and which are not
                if timeSeries in self.typicalPeriods:
                    dfTypicalPeriods = self.typicalPeriods[timeSeries]
                    AggregatedColumns = dfTimeSeries.columns.intersection(dfTypicalPeriods.columns)
                    NotAggregatedColumns = dfTimeSeries.columns.difference(dfTypicalPeriods.columns)
                    # aggregated columns
                    dfAggregatedTimeSeries[AggregatedColumns]       = self.typicalPeriods[timeSeries][AggregatedColumns]
                else:
                    NotAggregatedColumns = dfTimeSeries.columns
                # not aggregated columns
                dfAggregatedTimeSeries[NotAggregatedColumns]    = dfTimeSeries.iloc[0][NotAggregatedColumns]
                # reorder
                dfAggregatedTimeSeries.index.names              = [self.headerSetTimeSteps]
                dfAggregatedTimeSeries.columns.names            = _indexNames
                dfAggregatedTimeSeries                          = dfAggregatedTimeSeries.stack(_indexNames)
                dfAggregatedTimeSeries.index                    = dfAggregatedTimeSeries.index.reorder_levels(_indexNames + [self.headerSetTimeSteps])
                setattr(self.element,timeSeries,dfAggregatedTimeSeries)   
            
        # if time series of element were not aggregated, manual aggregation
        else:
            # setTimeSteps and duration
            if self.element.name in self.system["setCarriers"]: 
                setTimeSteps                            = self.dataInput.extractTimeSteps(self.inputPath)
                self.element.setTimeStepsCarrier        = setTimeSteps
                self.element.timeStepsCarrierDuration   = EnergySystem.calculateTimeStepDuration(self.element.setTimeStepsCarrier)
                self.element.orderTimeSteps             = np.concatenate([[timeStep]*self.element.timeStepsCarrierDuration[timeStep] for timeStep in self.element.timeStepsCarrierDuration])
            elif self.element.name in self.system["setTechnologies"]:
                setTimeSteps                            = self.dataInput.extractTimeSteps(self.inputPath,"operation")
                self.element.setTimeStepsOperation      = setTimeSteps
                self.element.timeStepsOperationDuration = EnergySystem.calculateTimeStepDuration(self.element.setTimeStepsOperation)
                self.element.orderTimeSteps             = np.concatenate([[timeStep]*self.element.timeStepsOperationDuration[timeStep] for timeStep in self.element.timeStepsOperationDuration])
            else:
                raise KeyError(f"{self.element.name} neither in setCarriers nor setTechnologies")
            for timeSeries in rawTimeSeries:
                # since the values are constant, it does not matter which time steps are extracted
                dfTimeSeries = rawTimeSeries[timeSeries].sort_index()
                setattr(self.element,timeSeries,dfTimeSeries.loc[(slice(None),setTimeSteps)])
