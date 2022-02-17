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
from sympy import true
import tsam.timeseriesaggregation as tsam

from model.objects.energy_system import EnergySystem
from model.objects.element import Element

class TimeSeriesAggregation():
    def __init__(self,element=None,inputPath=None,TSAOfSingleElement = True):
        self.setBaseTimeSteps   = EnergySystem.getEnergySystem().setBaseTimeSteps
        self.system             = EnergySystem.getSystem()
        self.analysis           = EnergySystem.getAnalysis()
        self.headerSetTimeSteps = self.analysis['headerDataInputs']["setTimeSteps"][0]
        if TSAOfSingleElement:
            self.element            = element
            self.inputPath          = inputPath
            self.dataInput          = element.dataInput
            # set aggregation object
            EnergySystem.setAggregationObjects(self.element,self)
            # get number of time steps
            self.getNumberOfTimeSteps()
            # select time series
            self.selectTimeSeries()
            # if raw nonconstant time series exist 
            if not self.dfTimeSeriesRaw.empty:
                # use multi time grid approach
                if self.system["multiGridTimeIndex"]:
                    # conduct time series aggregation
                    # substitute column names
                    self.substituteColumnNames(direction="flatten")
                    # run time series aggregation to create typical periods
                    self.runTimeSeriesAggregation()
                    # resubstitute column names
                    self.substituteColumnNames(direction="raise")
                    # set aggregated time series
                    self.setAggregatedTimeSeries()
                    # set aggregator indicators
                    self.setAggregationIndicators()
                else:
                    EnergySystem.setTimeSeriesRaw(self)
        # time series aggregation of entire input data set
        else:
            self.numberTypicalPeriods       = min(self.system["numberTimeStepsDefault"],np.size(self.system["setTimeSteps"]))
            self.numberTimeStepsPerPeriod   = 1
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
                

    def getNumberOfTimeSteps(self):
        """ this method extracts number of time steps for time series aggregation """
        if self.element.name in self.system["setCarriers"]: 
            typeOfTimeSteps = None
        elif self.element.name in self.system["setTechnologies"]:
            typeOfTimeSteps = "operation"
        else:
            raise KeyError(f"{self.element.name} neither in setCarriers nor setTechnologies")
        self.numberTypicalPeriods,self.numberTimeStepsPerPeriod = self.dataInput.extractTimeSteps(self.element.name,typeOfTimeSteps,getListOfTimeSteps=False)

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

    def selectTimeSeriesOfAllElements(self):
        """ this method retrieves the raw time series for the aggregation of all input data sets. Only in single time grid approach! """
        _dictRawTimeSeries              = {}
        for element in Element.getAllElements():
            _rawTimeSeries = EnergySystem.getTimeSeriesRaw(element)
            if _rawTimeSeries is not None:
                _dictRawTimeSeries[element.name] = _rawTimeSeries
        self.dfTimeSeriesRaw = pd.concat(_dictRawTimeSeries.values(),axis=1,keys=_dictRawTimeSeries.keys())

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
        # if not full time series
        if self.numberTypicalPeriods*self.numberTimeStepsPerPeriod != np.size(self.system["setTimeSteps"]):
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
            self.setTimeSteps       = self.aggregation.clusterPeriodIdx
            self.timeStepsDuration  = self.aggregation.clusterPeriodNoOccur
            self.orderTimeSteps     = self.aggregation.clusterOrder
        # if full time series, use input values 
        else:
            self.typicalPeriods     = self.dfTimeSeriesRaw
            if self.system["multiGridTimeIndex"]:
                if self.element.name in self.system["setTechnologies"]:
                    self.setTimeSteps   = self.element.dataInput.extractTimeSteps(self.element.name,"operation")
                else:
                    self.setTimeSteps   = self.element.dataInput.extractTimeSteps(self.element.name)
            else:
                self.setTimeSteps       = self.system["setTimeSteps"]
            self.timeStepsDuration  = EnergySystem.calculateTimeStepDuration(self.setTimeSteps)
            self.orderTimeSteps     = np.concatenate([[timeStep]*self.timeStepsDuration[timeStep] for timeStep in self.timeStepsDuration])
    
    def setAggregatedTimeSeries(self):
        """ this method sets the aggregated time series and sets the necessary attributes"""
        rawTimeSeries       = getattr(self.element,"rawTimeSeries")
        # setTimeSteps and duration
        if self.element.name in self.system["setCarriers"]: 
            self.element.setTimeStepsCarrier        = list(self.setTimeSteps)
            self.element.timeStepsCarrierDuration   = self.timeStepsDuration
            self.element.orderTimeSteps             = self.orderTimeSteps
        elif self.element.name in self.system["setTechnologies"]:
            self.element.setTimeStepsOperation      = list(self.setTimeSteps)
            self.element.timeStepsOperationDuration = self.timeStepsDuration
            self.element.orderTimeSteps             = self.orderTimeSteps
        else:
            raise KeyError(f"{self.element.name} neither in setCarriers nor setTechnologies")
        # iterate through raw time series
        for timeSeries in rawTimeSeries:
            _indexNames = list(rawTimeSeries[timeSeries].index.names)
            _indexNames.remove(self.headerSetTimeSteps)
            dfTimeSeries = rawTimeSeries[timeSeries].unstack(level = _indexNames)
            
            dfAggregatedTimeSeries = pd.DataFrame(index=self.setTimeSteps,columns=dfTimeSeries.columns)
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

    def setAggregatedTimeSeriesOfAllElements(self):
        """ this method sets the aggregated time series and sets the necessary attributes after the aggregation to a single time grid """

        for element in Element.getAllElements():
            rawTimeSeries = getattr(element,"rawTimeSeries")
            # setTimeSteps and duration
            if element.name in self.system["setCarriers"]: 
                element.setTimeStepsCarrier        = list(self.setTimeSteps)
                element.timeStepsCarrierDuration   = self.timeStepsDuration
                element.orderTimeSteps             = self.orderTimeSteps
            elif element.name in self.system["setTechnologies"]:
                element.setTimeStepsOperation      = list(self.setTimeSteps)
                element.timeStepsOperationDuration = self.timeStepsDuration
                element.orderTimeSteps             = self.orderTimeSteps
            else:
                raise KeyError(f"{element.name} neither in setCarriers nor setTechnologies")
            # iterate through raw time series
            for timeSeries in rawTimeSeries:
                _indexNames = list(rawTimeSeries[timeSeries].index.names)
                _indexNames.remove(self.headerSetTimeSteps)
                dfTimeSeries = rawTimeSeries[timeSeries].unstack(level = _indexNames)
                
                dfAggregatedTimeSeries = pd.DataFrame(index=self.setTimeSteps,columns=dfTimeSeries.columns)
                # columns which are in aggregated time series and which are not
                if element.name in self.typicalPeriods and timeSeries in self.typicalPeriods[element.name]:
                    dfTypicalPeriods = self.typicalPeriods[element.name,timeSeries]
                    AggregatedColumns = dfTimeSeries.columns.intersection(dfTypicalPeriods.columns)
                    NotAggregatedColumns = dfTimeSeries.columns.difference(dfTypicalPeriods.columns)
                    # aggregated columns
                    dfAggregatedTimeSeries[AggregatedColumns]       = self.typicalPeriods[element.name,timeSeries][AggregatedColumns]
                else:
                    NotAggregatedColumns = dfTimeSeries.columns
                # not aggregated columns
                dfAggregatedTimeSeries[NotAggregatedColumns]    = dfTimeSeries.iloc[0][NotAggregatedColumns]
                # reorder
                dfAggregatedTimeSeries.index.names              = [self.headerSetTimeSteps]
                dfAggregatedTimeSeries.columns.names            = _indexNames
                dfAggregatedTimeSeries                          = dfAggregatedTimeSeries.stack(_indexNames)
                dfAggregatedTimeSeries.index                    = dfAggregatedTimeSeries.index.reorder_levels(_indexNames + [self.headerSetTimeSteps])
                setattr(element,timeSeries,dfAggregatedTimeSeries)   
                # set aggregator indicators
                self.setAggregationIndicators(manualElement=element)

    def manuallyAggregateElement(self):
        """ this method manually aggregates elements which are not aggregated by the automatic aggregation method above """
        if self.element.name in self.system["setTechnologies"]:
            # get corresponding carriers
            if self.element in self.system["setConversionTechnologies"]:
                carriersOfTechnology = self.element.inputCarrier + self.element.outputCarrier
            else:
                carriersOfTechnology = self.element.referenceCarrier
            # get aggregated carriers
            listOrderAggregatedCarriers = []
            for _carrier in carriersOfTechnology:
                if Element.getElement(_carrier).isAggregated():
                    listOrderAggregatedCarriers.append(Element.getAttributeOfSpecificElement(_carrier,"orderTimeSteps"))
            # check if any carrier already aggregated
            if listOrderAggregatedCarriers:
                # set time steps of carrier on time steps of carriers
                # get combined time steps, duration and order
                setTimeStepsRaw, timeStepsDurationRaw, orderTimeStepsRaw = self.uniqueTimeStepsInMultigrid(listOrderAggregatedCarriers)   
                # if more timesteps demanded in not aggregated technology than combined in carriers
                if self.numberTimeStepsPerPeriod*self.numberTypicalPeriods >= len(setTimeStepsRaw):
                    if self.numberTimeStepsPerPeriod*self.numberTypicalPeriods > len(setTimeStepsRaw):
                        logging.warning(f"Requested number of time steps ({self.numberTypicalPeriods}/{self.numberTimeStepsPerPeriod}) of unaggregated technology {self.element.name} is greater than aggregated time steps in carriers ({len(setTimeStepsRaw)}). Restrict to time steps of carrier")
                    self.element.setTimeStepsOperation      = setTimeStepsRaw
                    self.element.timeStepsOperationDuration = timeStepsDurationRaw
                    self.element.orderTimeSteps             = orderTimeStepsRaw
                    for timeSeries in self.element.rawTimeSeries:
                        dfTimeSeries                        = self.element.rawTimeSeries[timeSeries].loc[(slice(None),setTimeStepsRaw)]
                        # save attribute
                        setattr(self.element,timeSeries,dfTimeSeries)
                # select subset of time steps, try to gather raw time steps to equal portions (cluster)
                else:
                    numberClusters      = self.numberTimeStepsPerPeriod*self.numberTypicalPeriods
                    # sort raw time steps by duration
                    sortedTimeStepsRaw  = dict(sorted(timeStepsDurationRaw.items(), key=lambda item: item[1],reverse=True))
                    idxSortedTimeSteps  = list(sortedTimeStepsRaw.keys())
                    counterTimeSteps    = 0
                    # current average duration per cluster, subsequently reduced
                    durationPerCluster  = sum(sortedTimeStepsRaw.values())/numberClusters
                    timeStepsInCluster  = {}
                    # initialize attributes
                    setClusters         = []
                    clusterDuration     = {}
                    orderTimeSteps      = np.zeros(np.size(self.system["setTimeSteps"])).astype(int)
                    for cluster in range(0,numberClusters):
                        # append cluster to attribute
                        setClusters.append(cluster)
                        _durationCurrentCluster         = 0
                        timeStepsInCluster[cluster]     = []
                        # iterate through time steps until duration of cluster exceeds average duration or no time steps left
                        while _durationCurrentCluster <= durationPerCluster and counterTimeSteps < len(setTimeStepsRaw):
                            timeStepsInCluster[cluster].append(idxSortedTimeSteps[counterTimeSteps])
                            # add to duration of current cluster
                            _durationCurrentCluster     += sortedTimeStepsRaw[idxSortedTimeSteps[counterTimeSteps]]
                            # remove time step from sorted dict
                            sortedTimeStepsRaw.pop(idxSortedTimeSteps[counterTimeSteps])
                            counterTimeSteps            += 1
                        clusterDuration[cluster]        = _durationCurrentCluster
                        numberClusters                  -= 1
                        if numberClusters != 0:
                            durationPerCluster  = sum(sortedTimeStepsRaw.values())/numberClusters
                        else:
                            durationPerCluster  = sum(sortedTimeStepsRaw.values())
                        # set index of time steps in order to cluster
                        orderTimeSteps[np.argwhere(np.isin(orderTimeStepsRaw,timeStepsInCluster[cluster]))] = cluster
                    # set attributes
                    self.element.setTimeStepsOperation      = setClusters
                    self.element.timeStepsOperationDuration = clusterDuration
                    self.element.orderTimeSteps             = orderTimeSteps
                    for timeSeries in self.element.rawTimeSeries:
                        dfTimeSeries                        = self.element.rawTimeSeries[timeSeries].loc[(slice(None),timeStepsInCluster)]
                        # save attribute
                        setattr(self.element,timeSeries,dfTimeSeries)
            else:
                # create equidistant time steps
                raise NotImplementedError("Manual aggregation of technologies where no related carrier is aggregated is not yet implemented!")

            # set aggregator indicators
            self.setAggregationIndicators()
        # carrier that is not yet aggregated
        else:
            # conduct calculation of carrier time steps as before
            self.calculateTimeStepsCarrier()

    def calculateTimeStepsCarrier(self):
        """ calculates the necessary time steps of carrier. Carrier must always have highest resolution of all connected technologies. 
        Can have even higher resolution"""
        setTimeHeaders          = EnergySystem.getAnalysis()["headerDataInputs"]["setTimeSteps"]
        # if carrier is already aggregated
        if self.element.isAggregated():
            orderTimeStepsRaw    = self.element.orderTimeSteps
            # if orderTimeStepsRaw not None
            listOrderTimeSteps  = [orderTimeStepsRaw]
        else:
            listOrderTimeSteps  = []
        # get technologies of carrier
        technologiesCarrier     = EnergySystem.getTechnologyOfCarrier(self.element.name)
        # if any technologies of carriers are aggregated 
        if technologiesCarrier: 
            # iterate through technologies and extend listOrderTimeSteps
            for technology in technologiesCarrier:
                listOrderTimeSteps.append(Element.getAttributeOfSpecificElement(technology,"orderTimeSteps"))
            # get combined time steps, duration and order
            setTimeStepsCarrier, timeStepsCarrierDuration, orderTimeStepsCarrier = self.uniqueTimeStepsInMultigrid(listOrderTimeSteps)
            # set attributes
            self.element.setTimeStepsEnergyBalance      = setTimeStepsCarrier
            self.element.timeStepsEnergyBalanceDuration = timeStepsCarrierDuration
            self.element.orderTimeStepsEnergyBalance    = orderTimeStepsCarrier
            # if carrier previously not aggregated
            if not self.element.isAggregated():
                # iterate through raw time series data and conduct "manual" time series aggregation
                for timeSeries in self.element.rawTimeSeries:
                    dfTimeSeries                        = self.element.rawTimeSeries[timeSeries].loc[(slice(None),setTimeStepsCarrier)]
                    # save attribute
                    setattr(self.element,timeSeries,dfTimeSeries)
                self.element.setTimeStepsCarrier        = setTimeStepsCarrier
                self.element.timeStepsCarrierDuration   = timeStepsCarrierDuration
                self.element.orderTimeSteps             = orderTimeStepsCarrier
            # set aggregation indicator
            self.setAggregationIndicators(setEnergyBalanceIndicator=True)

    def uniqueTimeStepsInMultigrid(self,listOrderTimeSteps):
        """ this method returns the unique time steps of multiple time grids """
        orderTimeSteps              = np.zeros(np.size(self.system["setTimeSteps"])).astype(int)
        combinedOrderTimeSteps      = np.vstack(listOrderTimeSteps)
        uniqueCombinedTimeSteps, countCombinedTimeSteps = np.unique(combinedOrderTimeSteps,axis=1,return_counts=True)
        setTimeSteps                = []
        timeStepsDuration           = {}
        for idxUniqueTimeStep, countUniqueTimeStep in enumerate(countCombinedTimeSteps):
            setTimeSteps.append(idxUniqueTimeStep)
            timeStepsDuration[idxUniqueTimeStep] = countUniqueTimeStep
            uniqueTimeStep                      = uniqueCombinedTimeSteps[:,idxUniqueTimeStep]
            idxInInput                          = np.argwhere(np.all(combinedOrderTimeSteps.T == uniqueTimeStep, axis=1))
            # fill new order time steps 
            orderTimeSteps[idxInInput]          = idxUniqueTimeStep
        return setTimeSteps, timeStepsDuration, orderTimeSteps

    def setAggregationIndicators(self,setEnergyBalanceIndicator = False,manualElement = None):
        """ this method sets the indicators that element is aggregated """
        if manualElement:
            element = manualElement
        else:
            element = self.element
        # add order of time steps to Energy System
        EnergySystem.setOrderTimeSteps(element.name,element.orderTimeSteps,timeStepType="operation") 
        # if energy balance indicator is set as well, save order of time steps in energy balance as well
        if setEnergyBalanceIndicator:
            EnergySystem.setOrderTimeSteps(element.name+"EnergyBalance",element.orderTimeStepsEnergyBalance,timeStepType="operation") 
        # if technology, add to technologyOfCarrier list
        if element.name in self.system["setTechnologies"]:
            if element.name in self.system["setConversionTechnologies"]:
                EnergySystem.setTechnologyOfCarrier(element.name,element.inputCarrier + element.outputCarrier)
            else:
                EnergySystem.setTechnologyOfCarrier(element.name,element.referenceCarrier)
                if element.name in self.system["setStorageTechnologies"]:
                    # calculate time steps of storage levels
                    element.calculateTimeStepsStorageLevel(EnergySystem.getPaths()["setStorageTechnologies"][element.name]["folder"])
        # set the aggregation status of element to true
        element.setAggregated()