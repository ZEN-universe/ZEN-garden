"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to extract the input data from the provided input files
==========================================================================================================================================================================="""
import copy
import os
import logging
import warnings
import math
import numpy  as np
import pandas as pd
from scipy.stats import linregress

class DataInput():

    def __init__(self,element,system,analysis,solver,energySystem,unitHandling):
        """ data input object to extract input data
        :param element: element for which data is extracted
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver 
        :param energySystem: instance of class <EnergySystem> to define energySystem
        :param unitHandling: instance of class <UnitHandling> to convert units """
        self.element        = element
        self.system         = system
        self.analysis       = analysis
        self.solver         = solver
        self.energySystem   = energySystem
        self.unitHandling   = unitHandling
        # extract folder path
        self.folderPath = getattr(self.element,"inputPath")

        # get names of indices
        self.indexNames     = {indexName: self.analysis['headerDataInputs'][indexName][0] for indexName in self.analysis['headerDataInputs']}

    def extractInputData(self,fileName,indexSets,column=None,timeSteps=None,scenario=""):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param fileName: name of selected file.
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param timeSteps: specific timeSteps of element
        :return dataDict: dictionary with attribute values """

        # generic time steps
        if not timeSteps:
            timeSteps = self.energySystem.setBaseTimeSteps
        # if time steps are the yearly time steps
        elif timeSteps == self.energySystem.setBaseTimeStepsYearly:
            self.extractYearlyVariation(fileName,indexSets,column)

        # if existing capacities and existing capacities not used
        if (fileName == "existingCapacity" or fileName == "existingCapacityEnergy") and not self.analysis["useExistingCapacities"]:
            dfOutput,*_ = self.createDefaultOutput(indexSets,column,fileName= fileName,timeSteps=timeSteps,manualDefaultValue=0,scenario=scenario)
            return dfOutput
        else:
            dfOutput, defaultValue, indexNameList = self.createDefaultOutput(indexSets,column,fileName =fileName, timeSteps= timeSteps,scenario=scenario)
        # set defaultName
        if column:
            defaultName = column
        else:
            defaultName = fileName
        # read input file
        dfInput = self.readInputData(fileName+scenario)

        assert(dfInput is not None or defaultValue is not None), f"input file for attribute {defaultName} could not be imported and no default value is given."
        if dfInput is not None and not dfInput.empty:
            dfOutput = self.extractGeneralInputData(dfInput,dfOutput,fileName,indexNameList,column,defaultValue)
        # save parameter values for analysis of numerics
        self.saveValuesOfAttribute(dfOutput=dfOutput,fileName=defaultName)
        return dfOutput

    def extractGeneralInputData(self,dfInput,dfOutput,fileName,indexNameList,column,defaultValue):
        """ fills dfOutput with data from dfInput
        :param dfInput: raw input dataframe
        :param dfOutput: empty output dataframe, only filled with defaultValue 
        :param fileName: name of selected file
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param defaultValue: default for dataframe
        :return dfOutput: filled output dataframe """

        # select and drop scenario
        assert dfInput.columns is not None, f"Input file '{fileName}' has no columns"
        assert self.indexNames["setScenarios"] not in dfInput.columns, f"the index '{self.indexNames['setScenarios']}' is depreciated, but still found in input file '{fileName}'"
        # set index by indexNameList
        missingIndex = list(set(indexNameList) - set(indexNameList).intersection(set(dfInput.columns)))
        assert len(missingIndex) <= 1, f"More than one the requested index sets ({missingIndex}) are missing from input file for {fileName}"

        # no indices missing
        if len(missingIndex) == 0:
            dfInput = DataInput.extractFromInputWithoutMissingIndex(dfInput,indexNameList,column,fileName)
        else:
            missingIndex = missingIndex[0]
            # check if special case of existing Technology
            if "existingTechnology" in missingIndex:
                if column:
                    defaultName = column
                else:
                    defaultName = fileName
                dfOutput = DataInput.extractFromInputForExistingCapacities(dfInput,dfOutput,indexNameList,defaultName,missingIndex)
                if isinstance(defaultValue,dict):
                    dfOutput = dfOutput * defaultValue["multiplier"]
                return dfOutput
            # index missing
            else:
                dfInput = DataInput.extractFromInputWithMissingIndex(dfInput,dfOutput,copy.deepcopy(indexNameList),column,fileName,missingIndex)

        # apply multiplier to input data
        dfInput     = dfInput * defaultValue["multiplier"]
        # delete nans
        dfInput     = dfInput.dropna()

        # get common index of dfOutput and dfInput
        if not isinstance(dfInput.index, pd.MultiIndex):
            indexList               = dfInput.index.to_list()
            if len(indexList) == 1:
                indexMultiIndex     = pd.MultiIndex.from_tuples([(indexList[0],)], names=[dfInput.index.name])
            else:
                indexMultiIndex     = pd.MultiIndex.from_product([indexList], names=[dfInput.index.name])
            dfInput                 = pd.Series(index=indexMultiIndex, data=dfInput.to_list())
        commonIndex                 = dfOutput.index.intersection(dfInput.index)
        assert defaultValue is not None or len(commonIndex) == len(dfOutput.index), f"Input for {fileName} does not provide entire dataset and no default given in attributes.csv"
        dfOutput.loc[commonIndex]   = dfInput.loc[commonIndex]
        return dfOutput

    def readInputData(self,inputFileName):
        """ reads input data and returns raw input dataframe
        :param inputFileName: name of selected file
        :return dfInput: pd.DataFrame with input data """

        # append .csv suffix
        inputFileName += ".csv"

        # select data
        fileNames = os.listdir(self.folderPath)
        if inputFileName in fileNames:
            dfInput = pd.read_csv(self.folderPath+inputFileName, header=0, index_col=None)
            return dfInput
        else:
            return None

    def extractAttributeData(self,attributeName,skipWarning = False,scenario=""):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param attributeName: name of selected attribute
        :param skipWarning: boolean to indicate if "Default" warning is skipped
        :return attributeValue: attribute value """
        filename = "attributes"
        dfInput  = self.readInputData(filename+scenario)
        if dfInput is not None:
            dfInput = dfInput.set_index("index").squeeze(axis=1)
            name    = self.adaptAttributeName(attributeName, dfInput, skipWarning)
        if dfInput is None or name is None:
            dfInput = self.readInputData(filename)
            if dfInput is not None:
                dfInput = dfInput.set_index("index").squeeze(axis=1)
            else:
                return None
        attributeName = self.adaptAttributeName(attributeName,dfInput,skipWarning)
        if attributeName is not None:
            # get attribute
            attributeValue = dfInput.loc[attributeName, "value"]
            multiplier = self.unitHandling.getUnitMultiplier(dfInput.loc[attributeName, "unit"])
            try:
                attribute = {"value": float(attributeValue) * multiplier, "multiplier": multiplier}
                return attribute
            except:
                return attributeValue
        else:
            return None

    def adaptAttributeName(self,attributeName,dfInput,skipWarning=False):
        """ check if attribute in index"""
        if attributeName + "Default" not in dfInput.index:
            if attributeName not in dfInput.index:
                return None
            elif not skipWarning:
                warnings.warn(
                    f"Attribute names without 'Default' suffix will be deprecated. \nChange for {attributeName} of attributes in path {self.folderPath}",
                    FutureWarning)
        else:
            attributeName = attributeName + "Default"
        return attributeName

    def extractYearlyVariation(self,fileName,indexSets,column):
        """ reads the yearly variation of a time dependent quantity
        :param self.folderPath: path to input files
        :param fileName: name of selected file.
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        """
        # remove intrayearly time steps from index set and add interyearly time steps
        _indexSets = copy.deepcopy(indexSets)
        _indexSets.remove("setTimeSteps")
        _indexSets.append("setTimeStepsYearly")
        # add YearlyVariation to fileName
        fileName  += "YearlyVariation"
        # read input data
        dfInput         = self.readInputData(fileName)
        if dfInput is not None:
            if column is not None and column not in dfInput:
                return
            dfOutput, defaultValue, indexNameList = self.createDefaultOutput(_indexSets,column,fileName = fileName, manualDefaultValue=1)
            # set yearlyVariation attribute to dfOutput
            if column:
                _selectedColumn         = column
                _nameYearlyVariation    = column+"YearlyVariation"
            else:
                _selectedColumn         = None
                _nameYearlyVariation    = fileName
            dfOutput = self.extractGeneralInputData(dfInput, dfOutput, fileName, indexNameList, _selectedColumn,defaultValue)
            setattr(self,_nameYearlyVariation,dfOutput)

    def extractLocations(self,extractNodes = True):
        """ reads input data to extract nodes or edges.
        :param extractNodes: boolean to switch between nodes and edges """
        if extractNodes:
            setNodesConfig  = self.system["setNodes"]
            setNodesInput   = self.readInputData("setNodes")["node"]
            # if no nodes specified in system, use all nodes
            if len(setNodesConfig) == 0 and not setNodesInput.empty:
                self.system["setNodes"] = setNodesInput
                setNodesConfig          = setNodesInput
            else:
                assert len(setNodesConfig) > 1, f"ZENx is a spatially distributed model. Please specify at least 2 nodes."
                _missingNodes   = list(set(setNodesConfig).difference(setNodesInput))
                assert len(_missingNodes) == 0, f"The nodes {_missingNodes} were declared in the config but do not exist in the input file {self.folderPath+'setNodes'}"
            setNodesConfig.sort()
            return setNodesConfig
        else:
            setEdgesInput = self.readInputData("setEdges")
            if setEdgesInput is not None:
                setEdges        = setEdgesInput[(setEdgesInput["nodeFrom"].isin(self.energySystem.setNodes)) & (setEdgesInput["nodeTo"].isin(self.energySystem.setNodes))]
                setEdges        = setEdges.set_index("edge")
                return setEdges
            else:
                return None

    def extractNumberTimeSteps(self):
        """ reads input data and returns number of typical periods and time steps per period for each technology and carrier
        :return dictNumberOfTimeSteps: number of typical periods for each technology """
        # select data
        fileName    = "setTimeSteps"
        dfInput     = self.readInputData(fileName)
        dfInput     = dfInput.set_index(["element","typeTimeStep"])
        # create empty dictNumberOfTimeSteps
        dictNumberOfTimeSteps = {}
        # iterate through investment time steps of technologies
        typeTimeStep = "invest"
        for technology in self.energySystem.setTechnologies:
            assert technology in dfInput.index.get_level_values("element"), f"Technology {technology} is not in {fileName}.csv"
            dictNumberOfTimeSteps[technology] = {}
            assert (technology,typeTimeStep) in dfInput.index, f"Type of time step <{typeTimeStep} for technology {technology} is not in {fileName}.csv"
            dictNumberOfTimeSteps[technology][typeTimeStep] = dfInput.loc[(technology,typeTimeStep)].squeeze()
        # iterate through carriers 
        for carrier in self.energySystem.setCarriers:
            assert carrier in dfInput.index.get_level_values("element"), f"Carrier {carrier} is not in {fileName}.csv"
            dictNumberOfTimeSteps[carrier] = {None: dfInput.loc[carrier].squeeze()}
        # add yearly time steps
        dictNumberOfTimeSteps[None] = {"yearly": self.system["optimizedYears"]}

        # limit number of periods to base time steps of system
        for element in dictNumberOfTimeSteps:
            # if yearly time steps
            if element is None:
                continue

            numberTypicalPeriods                = dictNumberOfTimeSteps[element][typeTimeStep]
            numberTypicalPeriodsTotal           = numberTypicalPeriods*dictNumberOfTimeSteps[None]["yearly"]
            if int(numberTypicalPeriodsTotal)   != numberTypicalPeriodsTotal:
                logging.warning(f"The requested invest time steps per year ({numberTypicalPeriods}) of {element} do not evaluate to an integer for the entire time horizon ({numberTypicalPeriodsTotal}). Rounded up.")
                numberTypicalPeriodsTotal                   = np.ceil(numberTypicalPeriodsTotal)
            dictNumberOfTimeSteps[element][typeTimeStep]    = int(numberTypicalPeriodsTotal)

        return dictNumberOfTimeSteps

    def extractTimeSteps(self,elementName=None,typeOfTimeSteps=None,getListOfTimeSteps=True):
        """ reads input data and returns range of time steps 
        :param elementName: name of element
        :param typeOfTimeSteps: type of time steps (invest, operational). If None, type column does not exist
        :param getListOfTimeSteps: boolean if list of time steps returned
        :return listOfTimeSteps: list of time steps """
        numberTypicalPeriods = self.energySystem.dictNumberOfTimeSteps[elementName][typeOfTimeSteps]
        if getListOfTimeSteps:
            # create range of time steps 
            listOfTimeSteps = list(range(0,numberTypicalPeriods))
            return listOfTimeSteps
        else:
            return numberTypicalPeriods

    def extractConversionCarriers(self):
        """ reads input data and extracts conversion carriers
        :param self.folderPath: path to input files
        :return carrierDict: dictionary with input and output carriers of technology """
        carrierDict = {}
        # get carriers
        for _carrierType in ["inputCarrier","outputCarrier"]:
            # TODO implement for multiple carriers
            _carrierString = self.extractAttributeData(_carrierType,skipWarning = True)
            if type(_carrierString) == str:
                _carrierList = _carrierString.strip().split(" ")
            else:
                _carrierList = []
            carrierDict[_carrierType] = _carrierList

        return carrierDict

    def extractSetExistingTechnologies(self, storageEnergy = False):
        """ reads input data and creates setExistingCapacity for each technology
        :param storageEnergy: boolean if existing energy capacity of storage technology (instead of power)
        :return setExistingTechnologies: return set existing technologies"""
        if self.analysis["useExistingCapacities"]:
            if storageEnergy:
                _energyString = "Energy"
            else:
                _energyString = ""

            dfInput = self.readInputData(f"existingCapacity{_energyString}")
            if dfInput is None:
                return  [0]

            if self.element.name in self.system["setTransportTechnologies"]:
                location = "edge"
            else:
                location = "node"
            maxNodeCount = dfInput[location].value_counts().max()
            setExistingTechnologies = np.arange(0, maxNodeCount)
        else:
            setExistingTechnologies = np.array([0])

        return setExistingTechnologies

    def extractLifetimeExistingTechnology(self, fileName, indexSets):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param fileName:  name of selected file
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :return existingLifetimeDict: return existing capacity and existing lifetime """
        column   = "yearConstruction"
        dfOutput = pd.Series(index=self.element.existingCapacity.index,data=0)
        # if no existing capacities
        if not self.analysis["useExistingCapacities"]:
            return dfOutput

        if f"{fileName}.csv" in os.listdir(self.folderPath):
            indexList, indexNameList = self.constructIndexList(indexSets, None)
            dfInput                  = self.readInputData( fileName)
            # fill output dataframe
            dfOutput = self.extractGeneralInputData(dfInput, dfOutput, fileName, indexNameList, column, defaultValue = 0)
            # get reference year
            referenceYear            = self.system["referenceYear"]
            # calculate remaining lifetime
            dfOutput[dfOutput > 0]   = - referenceYear + dfOutput[dfOutput > 0] + self.element.lifetime

        return dfOutput

    def extractPWAData(self,variableType):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param variableType: technology approximation type
        :return PWADict: dictionary with PWA parameters """
        # attribute names
        if variableType == "Capex":
            _attributeName  = "capexSpecific"
        elif variableType == "ConverEfficiency":
            _attributeName  = "converEfficiency"
        else:
            raise KeyError(f"variable type {variableType} unknown.")
        _indexSets = ["setNodes", "setTimeSteps"]
        _timeSteps = self.element.setTimeStepsInvest
        # import all input data
        dfInputNonlinear    = self.readPWAFiles(variableType, fileType="nonlinear")
        dfInputBreakpoints  = self.readPWAFiles(variableType, fileType="breakpointsPWA")
        dfInputLinear       = self.readPWAFiles(variableType, fileType="linear")
        ifLinearExist       = self.ifAttributeExists(_attributeName)
        assert (dfInputNonlinear is not None and dfInputBreakpoints is not None) \
               or ifLinearExist \
               or dfInputLinear is not None, \
            f"Neither PWA nor linear data exist for {variableType} of {self.element.name}"
        # check if capexSpecific exists
        if (dfInputNonlinear is not None and dfInputBreakpoints is not None):
            # select data
            PWADict = {}
            # extract all data values
            nonlinearValues     = {}

            if variableType == "Capex":
                # make absolute capex
                dfInputNonlinear["capex"] = dfInputNonlinear["capex"]*dfInputNonlinear["capacity"]
            for column in dfInputNonlinear.columns:
                nonlinearValues[column] = dfInputNonlinear[column].to_list()

            # assert that breakpoint variable (x variable in nonlinear input)
            assert dfInputBreakpoints.columns[0] in dfInputNonlinear.columns, f"breakpoint variable for PWA '{dfInputBreakpoints.columns[0]}' is not in nonlinear variables [{dfInputNonlinear.columns}]"
            breakpointVariable = dfInputBreakpoints.columns[0]
            breakpoints = dfInputBreakpoints[breakpointVariable].to_list()

            PWADict[breakpointVariable] = breakpoints
            PWADict["PWAVariables"]     = [] # select only those variables that are modeled as PWA
            PWADict["bounds"]           = {} # save bounds of variables
            LinearDict                  = {}
            # min and max total capacity of technology
            minCapacityTech,maxCapacityTech = (0,min(max(self.element.capacityLimit.values),max(breakpoints)))
            for valueVariable in nonlinearValues:
                if valueVariable == breakpointVariable:
                    PWADict["bounds"][valueVariable] = (minCapacityTech,maxCapacityTech)
                else:
                    # conduct linear regress
                    linearRegressObject = linregress(nonlinearValues[breakpointVariable],nonlinearValues[valueVariable])
                    # calculate relative intercept (intercept/slope) if slope != 0
                    if linearRegressObject.slope != 0:
                        _relativeIntercept = np.abs(linearRegressObject.intercept/linearRegressObject.slope)
                    else:
                        _relativeIntercept = np.abs(linearRegressObject.intercept)
                    # check if to a reasonable degree linear
                    if _relativeIntercept <= self.solver["linearRegressionCheck"]["epsIntercept"] and linearRegressObject.rvalue >= self.solver["linearRegressionCheck"]["epsRvalue"]:
                        # model as linear function
                        slopeLinReg = linearRegressObject.slope
                        LinearDict[valueVariable] = self.createDefaultOutput(indexSets=_indexSets, column=column, timeSteps=_timeSteps,
                                                 manualDefaultValue=slopeLinReg)[0]
                    else:
                        # model as PWA function
                        PWADict[valueVariable] = list(np.interp(breakpoints,nonlinearValues[breakpointVariable],nonlinearValues[valueVariable]))
                        PWADict["PWAVariables"].append(valueVariable)
                        # save bounds
                        _valuesBetweenBounds = [PWADict[valueVariable][idxBreakpoint] for idxBreakpoint,breakpoint in enumerate(breakpoints) if breakpoint >= minCapacityTech and breakpoint <= maxCapacityTech]
                        _valuesBetweenBounds.extend(list(np.interp([minCapacityTech,maxCapacityTech],breakpoints,PWADict[valueVariable])))
                        PWADict["bounds"][valueVariable] = (min(_valuesBetweenBounds),max(_valuesBetweenBounds))
            # PWA
            if (len(PWADict["PWAVariables"]) > 0 and len(LinearDict) == 0):
                isPWA = True
                return PWADict, isPWA
            # linear
            elif len(LinearDict) > 0 and len(PWADict["PWAVariables"]) == 0:
                isPWA = False
                LinearDict              = pd.DataFrame.from_dict(LinearDict)
                LinearDict.columns.name = "carrier"
                LinearDict              = LinearDict.stack()
                _converEfficiencyLevels = [LinearDict.index.names[-1]] + LinearDict.index.names[:-1]
                LinearDict              = LinearDict.reorder_levels(_converEfficiencyLevels)
                return LinearDict,  isPWA
            # no dependent carrier
            elif len(nonlinearValues) == 1:
                isPWA = False
                return None, isPWA
            else:
                raise NotImplementedError(f"There are both linearly and nonlinearly modeled variables in {variableType} of {self.element.name}. Not yet implemented")
        # linear
        else:
            isPWA = False
            LinearDict = {}
            if variableType == "Capex":
                LinearDict["capex"] = self.extractInputData("capexSpecific", indexSets=_indexSets, timeSteps=_timeSteps)
                return LinearDict,isPWA
            else:
                _dependentCarrier = list(set(self.element.inputCarrier + self.element.outputCarrier).difference(self.element.referenceCarrier))
                # TODO implement for more than 1 carrier
                if _dependentCarrier == []:
                    return None, isPWA
                elif len(_dependentCarrier) == 1 and dfInputLinear is None:
                    LinearDict[_dependentCarrier[0]] = self.extractInputData(_attributeName, indexSets=_indexSets, timeSteps=_timeSteps)
                else:
                    dfOutput,defaultValue,indexNameList = self.createDefaultOutput(_indexSets, None, timeSteps=_timeSteps, manualDefaultValue=1)
                    assert (dfInputLinear is not None), f"input file for linearConverEfficiency could not be imported."
                    dfInputLinear = dfInputLinear.rename(columns={'year': 'time'})
                    for carrier in _dependentCarrier:
                        LinearDict[carrier]        = self.extractGeneralInputData(dfInputLinear, dfOutput, "linearConverEfficiency", indexNameList, carrier, defaultValue).copy(deep=True)
                LinearDict = pd.DataFrame.from_dict(LinearDict)
                LinearDict.columns.name = "carrier"
                LinearDict = LinearDict.stack()
                _converEfficiencyLevels = [LinearDict.index.names[-1]] + LinearDict.index.names[:-1]
                LinearDict = LinearDict.reorder_levels(_converEfficiencyLevels)
                return LinearDict,isPWA

    def readPWAFiles(self,variableType,fileType):
        """ reads PWA Files
        :param variableType: technology approximation type
        :param fileType: either breakpointsPWA, linear, or nonlinear
        :return dfInput: raw input file"""
        dfInput             = self.readInputData(fileType+variableType)
        if dfInput is not None:
            if "unit" in dfInput.values:
                columns = dfInput.iloc[-1][dfInput.iloc[-1] != "unit"].dropna().index
            else:
                columns = dfInput.columns
            dfInputUnits        = dfInput[columns].iloc[-1]
            dfInput             = dfInput.iloc[:-1]
            dfInputMultiplier   = dfInputUnits.apply(lambda unit: self.unitHandling.getUnitMultiplier(unit))
            #dfInput[columns]    = dfInput[columns].astype(float
            dfInput             = dfInput.apply(lambda column: pd.to_numeric(column, errors='coerce'))
            dfInput[columns]    = dfInput[columns] * dfInputMultiplier
        return dfInput

    def createDefaultOutput(self,indexSets,column,fileName=None,timeSteps=None,manualDefaultValue = None,scenario = ""):
        """ creates default output dataframe
        :param fileName: name of selected file.
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param timeSteps: specific timeSteps of element
        :param scenario: investigated scenario
        :param manualDefaultValue: if given, use manualDefaultValue instead of searching for default value in attributes.csv"""
        # select index
        indexList, indexNameList = self.constructIndexList(indexSets, timeSteps)
        # create pd.MultiIndex and select data
        if indexSets:
            indexMultiIndex = pd.MultiIndex.from_product(indexList, names=indexNameList)
        else:
            indexMultiIndex = pd.Index([0])
        if manualDefaultValue:
            defaultValue = {"value":manualDefaultValue,"multiplier":1}
            defaultName  = None
        else:
            # check if default value exists in attributes.csv, with or without "Default" Suffix
            if column:
                defaultName = column
            else:
                defaultName = fileName
            defaultValue = self.extractAttributeData(defaultName,scenario=scenario)

        # create output Series filled with default value
        if defaultValue is None:
            dfOutput = pd.Series(index=indexMultiIndex, dtype=float)
        else:
            dfOutput = pd.Series(index=indexMultiIndex, data=defaultValue["value"], dtype=float)
        # save unit of attribute of element converted to base unit
        self.saveUnitOfAttribute(defaultName,scenario)
        return dfOutput,defaultValue,indexNameList

    def saveUnitOfAttribute(self,fileName,scenario=""):
        """ saves the unit of an attribute, converted to the base unit """
        # if numerics analyzed
        if self.solver["analyzeNumerics"]:
            if fileName:
                dfInput = self.readInputData("attributes" + scenario).set_index("index").squeeze(axis=1)
                # get attribute
                attributeName = self.adaptAttributeName(fileName,dfInput)
                inputUnit = dfInput.loc[attributeName, "unit"]
                self.unitHandling.setBaseUnitCombination(inputUnit=inputUnit,attribute=(self.element.name,fileName))

    def saveValuesOfAttribute(self,dfOutput,fileName):
        """ saves the values of an attribute """
        # if numerics analyzed
        if self.solver["analyzeNumerics"]:
            if fileName:
                dfOutputReduced = dfOutput[(dfOutput != 0) & (dfOutput.abs() != np.inf)]
                if not dfOutputReduced.empty:
                    self.unitHandling.setAttributeValues(dfOutput= dfOutputReduced,attribute=(self.element.name,fileName))

    def constructIndexList(self,indexSets,timeSteps):
        """ constructs index list from index sets and returns list of indices and list of index names
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param timeSteps: specific timeSteps of element
        :return indexList: list of indices
        :return indexNameList: list of name of indices
        """
        indexList     = []
        indexNameList = []

        # add rest of indices
        for index in indexSets:
            indexNameList.append(self.indexNames[index])
            if index == "setTimeSteps" and timeSteps:
                indexList.append(timeSteps)
            elif index == "setExistingTechnologies":
                indexList.append(self.element.setExistingTechnologies)
            elif index in self.system:
                indexList.append(self.system[index])
            elif hasattr(self.energySystem,index):
                indexList.append(getattr(self.energySystem,index))
            else:
                raise AttributeError(f"Index '{index}' cannot be found.")
        return indexList,indexNameList

    def ifAttributeExists(self, fileName, column=None):
        """ checks if default value or timeseries of an attribute exists in the input data
        :param fileName: name of selected file
        :param column: select specific column
        """
        # check if default value exists
        if column:
            defaultName = column
        else:
            defaultName = fileName
        defaultValue = self.extractAttributeData(defaultName)

        if defaultValue is None or math.isnan(defaultValue["value"]): # if no default value exists or default value is nan
            _dfInput = self.readInputData(fileName)
            return (_dfInput is not None)
        elif defaultValue and not math.isnan(defaultValue["value"]): # if default value exists and is not nan
            return True
        else:
            return False

    @staticmethod
    def extractFromInputWithoutMissingIndex(dfInput,indexNameList,column,fileName):
        """ extracts the demanded values from Input dataframe and reformulates dataframe
        :param dfInput: raw input dataframe
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param fileName: name of selected file
        :return dfInput: reformulated input dataframe
        """
        dfInput = dfInput.set_index(indexNameList)
        if column:
            assert column in dfInput.columns, f"Requested column {column} not in columns {dfInput.columns.to_list()} of input file {fileName}"
            dfInput = dfInput[column]
        else:
            # check if only one column remaining
            assert len(dfInput.columns) == 1,f"Input file for {fileName} has more than one value column: {dfInput.columns.to_list()}"
            dfInput = dfInput.squeeze(axis=1)
        return dfInput

    @staticmethod
    def extractFromInputWithMissingIndex(dfInput,dfOutput, indexNameList, column, fileName,missingIndex):
        """ extracts the demanded values from Input dataframe and reformulates dataframe if the index is missing.
        Either, the missing index is the column of dfInput, or it is actually missing in dfInput.
        Then, the values in dfInput are extended to all missing index values.
        :param dfInput: raw input dataframe
        :param dfOutput: default output dataframe
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param fileName: name of selected file
        :param missingIndex: missing index in dfInput
        :return dfInput: reformulated input dataframe
        """
        indexNameList.remove(missingIndex)
        dfInput                 = dfInput.set_index(indexNameList)
        # missing index values
        requestedIndexValues    = set(dfOutput.index.get_level_values(missingIndex))
        # the missing index is the columns of dfInput
        _requestedIndexValuesInColumns  = requestedIndexValues.intersection(dfInput.columns)
        if _requestedIndexValuesInColumns:
            requestedIndexValues    = _requestedIndexValuesInColumns
            dfInput.columns         = dfInput.columns.set_names(missingIndex)
            dfInput                 = dfInput[list(requestedIndexValues)].stack()
            dfInput                 = dfInput.reorder_levels(dfOutput.index.names)
        # the missing index does not appear in dfInput
        # the values in dfInput are extended to all missing index values
        else:
            # logging.info(f"Missing index {missingIndex} detected in {fileName}. Input dataframe is extended by this index")
            _dfInputIndexTemp   = pd.MultiIndex.from_product([dfInput.index,requestedIndexValues],names=dfInput.index.names+[missingIndex])
            _dfInputTemp        = pd.Series(index=_dfInputIndexTemp, dtype=float)
            if column in dfInput.columns:
                dfInput = dfInput[column].loc[_dfInputIndexTemp.get_level_values(dfInput.index.names[0])].squeeze()
                # much slower than overwriting index:
                # dfInput         = _dfInputTemp.to_frame().apply(lambda row: dfInput.loc[row.name[0], column].squeeze(),axis=1)
            else:
                if isinstance(dfInput,pd.Series):
                    dfInput = dfInput.to_frame()
                if dfInput.shape[1] == 1:
                    dfInput         = dfInput.loc[_dfInputIndexTemp.get_level_values(dfInput.index.names[0])].squeeze()
                else:
                    assert _dfInputTemp.index.names[-1] != "time", f"Only works if columns contain time index and not for {_dfInputTemp.index.names[-1]}"
                    dfInput = _dfInputTemp.to_frame().apply(lambda row: dfInput.loc[row.name[0:-1],str(row.name[-1])],axis=1)
            dfInput.index = _dfInputTemp.index
            dfInput = dfInput.reorder_levels(order=dfOutput.index.names)
            if isinstance(dfInput,pd.DataFrame):
                dfInput = dfInput.squeeze()
        return dfInput

    @staticmethod
    def extractFromInputForExistingCapacities(dfInput,dfOutput, indexNameList, column, missingIndex):
        """ extracts the demanded values from input dataframe if extracting existing capacities
        :param dfInput: raw input dataframe
        :param dfOutput: default output dataframe
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param missingIndex: missing index in dfInput
        :return dfOutput: filled output dataframe
        """
        indexNameList.remove(missingIndex)
        dfInput = dfInput.set_index(indexNameList)
        setLocation = dfInput.index.unique()
        for location in setLocation:
            if location in dfOutput.index.get_level_values(indexNameList[0]):
                values = dfInput[column].loc[location].tolist()
                if isinstance(values, int) or isinstance(values, float):
                    index = [0]
                else:
                    index = list(range(len(values)))
                dfOutput.loc[location, index] = values
        return dfOutput
