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
import numpy  as np
import pandas as pd
from scipy.stats import linregress

class DataInput():

    def __init__(self,element,system,analysis,solver,energySystem = None):
        """ data input object to extract input data
        :param element: element for which data is extracted
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver 
        :param energySystem: instance of class <EnergySystem> to define energySystem """
        self.system         = system
        self.analysis       = analysis
        self.solver         = solver
        self.energySystem   = energySystem
        self.element        = element
        # get names of indices
        self.indexNames     = {indexName: self.analysis['headerDataInputs'][indexName][0] for indexName in self.analysis['headerDataInputs']}

    def readInputData(self,folderPath,manualFileName):
        """ reads input data and returns raw input dataframe
        :param folderPath: path to input files 
        :param manualFileName: name of selected file. If only one file in folder, not used
        :return dfInput: pd.DataFrame with input data 
        :return fileName: name of file """

        # get system attribute
        fileFormat  = self.analysis["fileFormat"]

        # select data
        fileNames = [fileName.split('.')[0] for fileName in os.listdir(folderPath) if (fileName.split('.')[-1] == fileFormat)]
        if manualFileName in fileNames:
            assert (manualFileName in fileNames or len(fileNames) == 1), "Selection of files was ambiguous. Select folder with single input file or select specific file by name"
            for fileName in fileNames:
                if len(fileNames) > 1 and fileName != manualFileName:
                    continue

                # table attributes
                dfInput = pd.read_csv(folderPath+fileName+'.'+fileFormat, header=0, index_col=None) 
                break 
            return dfInput,fileName
        else:
            return None, None
    
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
            # if index == "setEdges":
            #     indexList.append(self.energySystem.setEdges)
            if index == "setTimeSteps" and timeSteps:
                indexList.append(timeSteps)
            elif index == "setExistingTechnologies":
                indexList.append(self.element.setExistingTechnologies)
            elif index in self.system:
                indexList.append(self.system[index])
            elif hasattr(self.energySystem,index):
                indexList.append(getattr(self.energySystem,index))
        return indexList,indexNameList

    def createDefaultOutput(self, folderPath,manualFileName,indexSets,column,timeSteps=None,manualDefaultValue = None):
        """ creates default output dataframe
        :param folderPath: path to input files
        :param manualFileName: name of selected file. If only one file in folder, not used
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param timeSteps: specific timeSteps of element
        :param manualDefaultValue: if given, use manualDefaultValue instead of searching for default value in attributes.csv"""
        # select index
        indexList, indexNameList = self.constructIndexList(indexSets, timeSteps)
        # create pd.MultiIndex and select data
        indexMultiIndex = pd.MultiIndex.from_product(indexList, names=indexNameList)
        if manualDefaultValue:
            defaultValue = {"value":manualDefaultValue,"multiplier":1}
            defaultName = manualFileName
        else:
            # check if default value exists in attributes.csv, with or without "Default" Suffix
            if column:
                defaultName = column
            else:
                defaultName = manualFileName
            defaultValue = self.extractAttributeData(folderPath, defaultName)

        # create output Series filled with default value
        if defaultValue is None:
            dfOutput = pd.Series(index=indexMultiIndex, dtype=float)
        else:
            dfOutput = pd.Series(index=indexMultiIndex, data=defaultValue["value"], dtype=float)

        return dfOutput,defaultValue,indexMultiIndex,indexNameList,defaultName

    def extractInputData(self, folderPath,manualFileName,indexSets,column=None,timeSteps=[],transportTechnology=False):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files 
        :param manualFileName: name of selected file. If only one file in folder, not used
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param timeSteps: specific timeSteps of element
        :param transportTechnology: boolean if data extracted for transport technology
        :return dataDict: dictionary with attribute values """

        # generic time steps
        if not timeSteps:
            timeSteps = self.energySystem.setBaseTimeSteps
        # if time steps are the yearly time steps
        elif timeSteps == self.energySystem.setBaseTimeStepsYearly:
            self.extractYearlyVariation(folderPath,manualFileName,indexSets,column)

        # if existing capacities and existing capacities not used
        if manualFileName == "existingCapacity" and not self.analysis["useExistingCapacities"]:
            dfOutput,*_ = self.createDefaultOutput(folderPath,manualFileName,indexSets,column,timeSteps,manualDefaultValue=0)
            return dfOutput
        else:
            dfOutput, defaultValue, indexMultiIndex, indexNameList, defaultName = self.createDefaultOutput(folderPath,
                                                                                                           manualFileName,
                                                                                                           indexSets,
                                                                                                           column,
                                                                                                           timeSteps)
        # read input file
        dfInput,fileName = self.readInputData(folderPath,manualFileName)
        assert(dfInput is not None or defaultValue is not None), f"input file for attribute {defaultName} could not be imported and no default value is given."
        if dfInput is not None and not dfInput.empty:
            # if not extracted for transport technology
            if not transportTechnology:
                dfOutput = self.extractGeneralInputData(dfInput,dfOutput,fileName,indexNameList,column,defaultValue)
            else:
                dfOutput = self.extractTransportInputData(dfInput,dfOutput,fileName,indexMultiIndex,column,defaultValue)
        return dfOutput 
    
    def extractGeneralInputData(self,dfInput,dfOutput,fileName,indexNameList,column,defaultValue):
        """ fills dfOutput with data from dfInput with no new index creation (no transport technologies)
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
                dfOutput = DataInput.extractFromInputForExistingCapacities(dfInput,dfOutput,indexNameList,column,missingIndex)
                return dfOutput
            # index missing
            else:
                dfInput = DataInput.extractFromInputWithMissingIndex(dfInput,dfOutput,indexNameList,column,fileName,missingIndex)

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

    def extractTransportInputData(self, dfInput,dfOutput,fileName,indexMultiIndex,column,defaultValue):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param dfInput: raw input dataframe
        :param dfOutput: empty output dataframe, only filled with defaultValue
        :param fileName: name of selected file
        :param indexMultiIndex: multiIndex of dfOutput
        :param column: select specific column
        :param defaultValue: default for dataframe
        :return dfOutput: filled output dataframe """

        # preferably already edges as index
        if self.indexNames["setEdges"] in dfInput.columns:
            dfOutput = self.extractGeneralInputData(dfInput,dfOutput,fileName,indexNameList=list(indexMultiIndex.names),column=column,defaultValue=defaultValue)
        else:
            warnings.warn(f"The matrix representation of edges will be deprecated. Change file '{fileName}'",FutureWarning)
            dfInput = dfInput.set_index(self.indexNames['setNodes'])

            # apply multiplier to input data
            dfInput = dfInput * defaultValue["multiplier"]

            # fill dfOutput
            for index in indexMultiIndex:
                if isinstance(index,tuple):
                    _nodeFrom,_nodeTo = self.energySystem.setNodesOnEdges[index[0]]
                else:
                    _nodeFrom,_nodeTo = self.energySystem.setNodesOnEdges[index]
                if _nodeFrom in dfInput.index and _nodeTo in dfInput.columns and ~np.isnan(dfInput.loc[_nodeFrom,_nodeTo]):
                    dfOutput.loc[index] = dfInput.loc[_nodeFrom,_nodeTo]
        return dfOutput

    def extractYearlyVariation(self, folderPath,manualFileName,indexSets,column):
        """ reads the yearly variation of a time dependent quantity
        :param folderPath: path to input files
        :param manualFileName: name of selected file. If only one file in folder, not used
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        """
        # remove intrayearly time steps from index set and add interyearly time steps
        _indexSets = copy.deepcopy(indexSets)
        _indexSets.remove("setTimeSteps")
        _indexSets.append("setTimeStepsYearly")
        # add YearlyVariation to manualFileName
        manualFileName  += "YearlyVariation"
        # read input data
        _rawInputData   = self.readInputData(folderPath, manualFileName)
        if _rawInputData[0] is not None:
            if column is not None and column not in _rawInputData[0]:
                return
            dfOutput, defaultValue, indexMultiIndex, indexNameList, defaultName = self.createDefaultOutput(folderPath,
                                                                                                           manualFileName,
                                                                                                           _indexSets,
                                                                                                           column,
                                                                                                           manualDefaultValue=1)
            dfInput, fileName   = self.readInputData(folderPath, manualFileName)
            # set yearlyVariation attribute to dfOutput
            if column is not None:
                dfOutput = self.extractGeneralInputData(dfInput, dfOutput, fileName, indexNameList, column,
                                                        defaultValue)
                setattr(self,column+"YearlyVariation",dfOutput)
            else:
                dfOutput = self.extractGeneralInputData(dfInput, dfOutput, fileName, indexNameList, dfInput.columns[-1],
                                                        defaultValue)
                setattr(self,manualFileName,dfOutput)

    def extractAttributeData(self, folderPath,attributeName,skipWarning = False):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files
        :param attributeName: name of selected attribute
        :param skipWarning: boolean to indicate if "Default" warning is skipped
        :return attributeValue: attribute value """
        fileName    = "attributes.csv"

        if fileName not in os.listdir(folderPath):
            return None
        dfInput     = pd.read_csv(folderPath+fileName, header=0, index_col=None).set_index("index").squeeze(axis=1)

        # check if attribute in index
        if attributeName+"Default" not in dfInput.index:
            if attributeName not in dfInput.index:
                warnings.warn(
                    f"Attribute without default value will be deprecated. \nAdd default value for {attributeName} in attribute file in {folderPath}",
                    FutureWarning)
                return None
            elif not skipWarning:
                warnings.warn(
                    f"Attribute names without 'Default' suffix will be deprecated. \nChange for {attributeName} of attributes in path {folderPath}",
                    FutureWarning)
        else:
            attributeName = attributeName + "Default"

        # get attribute
        attributeValue = dfInput.loc[attributeName, "value"]
        multiplier = self.getUnitMultiplier(dfInput.loc[attributeName, "unit"])
        try:
            attribute = {"value": float(attributeValue) * multiplier, "multiplier": multiplier}
            return attribute
        except:
            return attributeValue

    def ifAttributeExists(self, folderPath, manualFileName, column=None):
        """ checks if default value or timeseries of an attribute exists in the input data
        :param folderPath: path to input files
        :param manualFileName: name of selected file. If only one file in folder, not used
        :param column: select specific column
        """

        # check if default value exists
        if column:
            defaultName = column
        else:
            defaultName = manualFileName
        defaultValue = self.extractAttributeData(folderPath, defaultName)

        # check if input file exists
        inputData = None
        if defaultValue is None:
            inputData, _ = self.readInputData(folderPath, manualFileName)

        if defaultValue is None and inputData is None:
            return False
        else:
            return True

    def extractLocations(self,extractNodes = True):
        """ reads input data to extract nodes or edges.
        :param extractNodes: boolean to switch between nodes and edges """
        folderPath = self.energySystem.getPaths()["setNodes"]["folder"]
        if extractNodes:
            setNodesConfig  = self.system["setNodes"]
            setNodesInput   = self.readInputData(folderPath,"setNodes")[0]["node"]
            _missingNodes   = list(set(setNodesConfig).difference(setNodesInput))
            assert len(_missingNodes) == 0, f"The nodes {_missingNodes} were declared in the config but do not exist in the input file {folderPath+'setNodes'}"
            return setNodesConfig
        else:
            setEdgesInput = self.readInputData(folderPath,"setEdges")[0]
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
        folderName  = "setTimeSteps"
        fileName    = "setTimeSteps"
        dfInput,_   = self.readInputData(self.energySystem.paths[folderName]["folder"],fileName)
        dfInput     = dfInput.set_index(["element","typeTimeStep"])
        # create empty dictNumberOfTimeSteps
        dictNumberOfTimeSteps = {}
        # iterate through investment time steps of technologies
        typeTimeStep = "invest"
        for technology in self.energySystem.setTechnologies:
            assert technology in dfInput.index.get_level_values("element"), f"Technology {technology} is not in {fileName}.{self.analysis['fileFormat']}"
            dictNumberOfTimeSteps[technology] = {}
            assert (technology,typeTimeStep) in dfInput.index, f"Type of time step <{typeTimeStep} for technology {technology} is not in {fileName}.{self.analysis['fileFormat']}"
            dictNumberOfTimeSteps[technology][typeTimeStep] = dfInput.loc[(technology,typeTimeStep)].squeeze()
        # iterate through carriers 
        for carrier in self.energySystem.setCarriers:
            assert carrier in dfInput.index.get_level_values("element"), f"Carrier {carrier} is not in {fileName}.{self.analysis['fileFormat']}"
            dictNumberOfTimeSteps[carrier] = {None: dfInput.loc[carrier].squeeze()}
        # add yearly time steps
        dictNumberOfTimeSteps[None] = {"yearly": self.system["optimizedYears"]}

        # limit number of periods to base time steps of system
        for element in dictNumberOfTimeSteps:
            # if yearly time steps
            if element is None:
                continue

            numberTypicalPeriods = dictNumberOfTimeSteps[element][typeTimeStep]
            numberTypicalPeriodsTotal = numberTypicalPeriods*self.system["optimizedYears"]
            if int(numberTypicalPeriodsTotal) != numberTypicalPeriodsTotal:
                logging.warning(f"The requested invest time steps per year ({numberTypicalPeriods}) of {element} do not evaluate to an integer for the entire time horizon ({numberTypicalPeriodsTotal}). Rounded up.")
                numberTypicalPeriodsTotal = np.ceil(numberTypicalPeriodsTotal)
            dictNumberOfTimeSteps[element][typeTimeStep] = int(numberTypicalPeriodsTotal)

        return dictNumberOfTimeSteps

    def extractTimeSteps(self,elementName=None,typeOfTimeSteps=None,getListOfTimeSteps=True):
        """ reads input data and returns range of time steps 
        :param folderPath: path to input files 
        :param typeOfTimeSteps: type of time steps (invest, operational). If None, type column does not exist
        :return listOfTimeSteps: list of time steps """
        numberTypicalPeriods = self.energySystem.dictNumberOfTimeSteps[elementName][typeOfTimeSteps]
        if getListOfTimeSteps:
            # create range of time steps 
            listOfTimeSteps = list(range(0,numberTypicalPeriods))
            return listOfTimeSteps
        else:
            return numberTypicalPeriods

    def extractConversionCarriers(self, folderPath):
        """ reads input data and extracts conversion carriers
        :param folderPath: path to input files 
        :return carrierDict: dictionary with input and output carriers of technology """
        carrierDict = {}
        # get carriers
        for _carrierType in ["inputCarrier","outputCarrier"]:
            # TODO implement for multiple carriers
            _carrierString = self.extractAttributeData(folderPath,_carrierType,skipWarning = True)
            if type(_carrierString) == str:
                _carrierList = _carrierString.strip().split(" ")
            else:
                _carrierList = []
            carrierDict[_carrierType] = _carrierList

        return carrierDict

    def extractSetExistingTechnologies(self, folderPath, transportTechnology=False,storageEnergy = False):
        """ reads input data and creates setExistingCapacity for each technology
        :param folderPath: path to input files
        :param transportTechnology: boolean if data extracted for transport technology
        :param storageEnergy: boolean if existing energy capacity of storage technology (instead of power)
        :return setExistingTechnologies: return set existing technologies"""
        if self.analysis["useExistingCapacities"]:
            if storageEnergy:
                _energyString = "Energy"
            else:
                _energyString = ""
            fileFormat = self.analysis["fileFormat"]

            if f"existingCapacity{_energyString}.{fileFormat}" not in os.listdir(folderPath):
                return [0]

            dfInput = pd.read_csv(folderPath + f"existingCapacity{_energyString}." + fileFormat, header=0, index_col=None)

            if not transportTechnology:
                location = "node"
            else:
                location = "edge"
            maxNodeCount = dfInput[location].value_counts().max()
            setExistingTechnologies = np.arange(0, maxNodeCount)
        else:
            setExistingTechnologies = np.array([0])

        return setExistingTechnologies

    def extractLifetimeExistingTechnology(self, folderPath, fileName, indexSets, tech, transportTechnology=False):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files
        :param tech: technology object
        :return existingLifetimeDict: return existing Capacity and existing Lifetime """
        fileFormat   = self.analysis["fileFormat"]
        column       = "yearConstruction"
        defaultValue = 0

        dfOutput    = pd.Series(index=tech.existingCapacity.index,data=0)
        # if no existing capacities
        if not self.analysis["useExistingCapacities"]:
            return dfOutput

        if f"{fileName}.{fileFormat}" in os.listdir(folderPath):
            indexList, indexNameList    = self.constructIndexList(indexSets, None)
            dfInput, fileName           = self.readInputData(folderPath, fileName)
            indexMultiIndex             = pd.MultiIndex.from_product(indexList, names=indexNameList)
            # if not extracted for transport technology
            if not transportTechnology:
                dfOutput = self.extractGeneralInputData(dfInput, dfOutput, fileName, indexNameList, column, defaultValue)
            else:
                dfOutput = self.extractTransportInputData(dfInput, dfOutput, fileName, indexMultiIndex,column, defaultValue)
            # get reference year
            referenceYear               = self.system["referenceYear"]
            # calculate remaining lifetime
            dfOutput[dfOutput > 0]      = -referenceYear + dfOutput[dfOutput > 0] + tech.lifetime

        return dfOutput

    def extractPWAData(self, folderPath,type,tech):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files
        :param type: technology approximation type
        :param tech: technology object
        :return PWADict: dictionary with PWA parameters """

        # get system attribute
        fileFormat = self.analysis["fileFormat"]
        # select data
        PWADict = {}
        assert type in self.analysis["nonlinearTechnologyApproximation"], f"{type} is not specified in analysis['nonlinearTechnologyApproximation']"

        # extract all data values
        nonlinearValues     = {}
        assert f"nonlinear{type}.{fileFormat}" in os.listdir(folderPath), f"File 'nonlinear{type}.{fileFormat}' does not exist in {folderPath}"
        dfInputNonlinear    = pd.read_csv(folderPath+"nonlinear" + type + '.'+fileFormat, header=0, index_col=None)
        dfInputUnits        = dfInputNonlinear.iloc[-1]
        dfInputMultiplier   = dfInputUnits.apply(lambda unit: self.getUnitMultiplier(unit))
        dfInputNonlinear    = dfInputNonlinear.iloc[:-1].astype(float)
        dfInputNonlinear    = dfInputNonlinear*dfInputMultiplier
        if type == "Capex":
            # make absolute capex
            dfInputNonlinear["capex"] = dfInputNonlinear["capex"]*dfInputNonlinear["capacity"]
        for column in dfInputNonlinear.columns:
            nonlinearValues[column] = dfInputNonlinear[column].to_list()
        # extract PWA breakpoints
        assert f"breakpointsPWA{type}.{fileFormat}" in os.listdir(folderPath), f"File 'breakpointsPWA{type}.{fileFormat}' does not exist in {folderPath}"
        # TODO devise better way to split string units
        dfInputBreakpoints          = pd.read_csv(folderPath+"breakpointsPWA" + type + '.'+fileFormat, header=0, index_col=None)
        dfInputBreakpointsUnits     = dfInputBreakpoints.iloc[-1]
        dfInputMultiplier           = dfInputBreakpointsUnits.apply(lambda unit: self.getUnitMultiplier(unit))
        dfInputBreakpoints          = dfInputBreakpoints.iloc[:-1].astype(float)
        dfInputBreakpoints          = dfInputBreakpoints*dfInputMultiplier
        # assert that breakpoint variable (x variable in nonlinear input)
        assert dfInputBreakpoints.columns[0] in dfInputNonlinear.columns, f"breakpoint variable for PWA '{dfInputBreakpoints.columns[0]}' is not in nonlinear variables [{dfInputNonlinear.columns}]"
        breakpointVariable = dfInputBreakpoints.columns[0]
        breakpoints = dfInputBreakpoints[breakpointVariable].to_list()

        PWADict[breakpointVariable] = breakpoints
        PWADict["PWAVariables"] = [] # select only those variables that are modeled as PWA
        PWADict["bounds"] = {} # save bounds of variables
        # min and max total capacity of technology
        minCapacityTech,maxCapacityTech = (0,min(max(tech.capacityLimit.values),max(breakpoints)))
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
                    PWADict[valueVariable] = linearRegressObject.slope
                    # save bounds
                    PWADict["bounds"][valueVariable] = (PWADict[valueVariable]*minCapacityTech,PWADict[valueVariable]*maxCapacityTech)
                else:
                    # model as PWA function
                    PWADict[valueVariable] = list(np.interp(breakpoints,nonlinearValues[breakpointVariable],nonlinearValues[valueVariable]))
                    PWADict["PWAVariables"].append(valueVariable)
                    # save bounds
                    _valuesBetweenBounds = [PWADict[valueVariable][idxBreakpoint] for idxBreakpoint,breakpoint in enumerate(breakpoints) if breakpoint >= minCapacityTech and breakpoint <= maxCapacityTech]
                    _valuesBetweenBounds.extend(list(np.interp([minCapacityTech,maxCapacityTech],breakpoints,PWADict[valueVariable])))
                    PWADict["bounds"][valueVariable] = (min(_valuesBetweenBounds),max(_valuesBetweenBounds))
        return PWADict

    def extractBaseUnits(self,folderPath):
        """ extracts base units of energy system
        :param folderPath: path to input files
        :return listBaseUnits: list of base units """
        listBaseUnits = pd.read_csv(folderPath +"/baseUnits.csv").squeeze().values.tolist()
        return listBaseUnits

    def getUnitMultiplier(self,inputUnit):
        """ calculates the multiplier for converting an inputUnit to the base units
        :param inputUnit: string of input unit
        :return multiplier: multiplication factor """
        ureg        = self.energySystem.ureg
        baseUnits   = self.energySystem.baseUnits
        dimMatrix   = self.energySystem.dimMatrix
        # if input unit is already in base units --> the input unit is base unit, multiplier = 1
        if inputUnit in baseUnits:
            return 1
        # if input unit is nan --> dimensionless
        elif type(inputUnit) != str and np.isnan(inputUnit):
            return 1
        else:
            # create dimensionality vector for inputUnit
            dimInput    = ureg.get_dimensionality(ureg(inputUnit))
            dimVector   = pd.Series(index=dimMatrix.index, data=0)
            _missingDim = set(dimInput.keys()).difference(dimVector.keys())
            assert len(_missingDim) == 0, f"No base unit defined for dimensionalities <{_missingDim}>"
            dimVector[list(dimInput.keys())] = list(dimInput.values())
            # calculate dimensionless combined unit (e.g., tons and kilotons)
            combinedUnit = ureg(inputUnit).units
            # if unit (with a different multiplier) is already in base units
            if dimMatrix.isin(dimVector).all(axis=0).any():
                _baseUnit       = ureg(dimMatrix.columns[dimMatrix.isin(dimVector).all(axis=0)][0])
                combinedUnit    *= _baseUnit**(-1)
            # if inverse of unit (with a different multiplier) is already in base units (e.g. 1/km and km)
            elif (dimMatrix*-1).isin(dimVector).all(axis=0).any():
                _baseUnit       = ureg(dimMatrix.columns[(dimMatrix*-1).isin(dimVector).all(axis=0)][0])
                combinedUnit    *= _baseUnit
            else:
                dimAnalysis         = self.energySystem.dimAnalysis
                # drop dependent units
                dimMatrixReduced    = dimMatrix.drop(dimAnalysis["dependentUnits"],axis=1)
                # solve system of linear equations
                combinationSolution = np.linalg.solve(dimMatrixReduced,dimVector)
                # check if only -1, 0, 1
                if DataInput.checkIfPosNegBoolean(combinationSolution):
                    # compose relevant units to dimensionless combined unit
                    for unit,power in zip(dimMatrixReduced.columns,combinationSolution):
                        combinedUnit *= ureg(unit)**(-1*power)
                else:
                    calculatedMultiplier = False
                    for unit, power in zip(dimMatrixReduced.columns, combinationSolution):
                        # try to substitute unit with power > 1 by a dependent unit
                        if np.abs(power) > 1:
                            # iterate through dependent units
                            for dependentUnit,dependentDim in zip(dimAnalysis["dependentUnits"],dimAnalysis["dependentDims"]):
                                idxUnitInMatrixReduced  = list(dimMatrixReduced.columns).index(unit)
                                # if the power of the unit is the same as of the dimensionality in the dependent unit
                                if np.abs(dependentDim[idxUnitInMatrixReduced]) == np.abs(power):
                                    dimMatrixReducedTemp                    = dimMatrixReduced.drop(unit,axis=1)
                                    dimMatrixReducedTemp[dependentUnit]     = dimMatrix[dependentUnit]
                                    combinationSolutionTemp                 = np.linalg.solve(dimMatrixReducedTemp, dimVector)
                                    if DataInput.checkIfPosNegBoolean(combinationSolutionTemp):
                                        # compose relevant units to dimensionless combined unit
                                        for unit, power in zip(dimMatrixReducedTemp.columns, combinationSolutionTemp):
                                            combinedUnit        *= ureg(unit) ** (-1 * power)
                                        calculatedMultiplier    = True
                                        break
                    assert calculatedMultiplier, f"Cannot establish base unit conversion for {inputUnit} from base units {baseUnits.keys()}"
            # magnitude of combined unit is multiplier
            multiplier = combinedUnit.to_base_units().magnitude
            # round to decimal points
            return round(multiplier,self.solver["roundingDecimalPoints"])

    @classmethod
    def checkIfPosNegBoolean(cls, array,axis=None):
        """ checks if the array has only positive or negative booleans (-1,0,1)
        :param array: numeric numpy array
        :return isPosNegBoolean """
        if axis:
            isPosNegBoolean = np.apply_along_axis(lambda row: np.array_equal(np.abs(row), np.abs(row).astype(bool)),1,array).any()
        else:
            isPosNegBoolean = np.array_equal(np.abs(array), np.abs(array).astype(bool))
        return isPosNegBoolean

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
            assert column in dfInput.columns,\
                f"Requested column {column} not in columns {dfInput.columns.to_list()} of input file {fileName}"
            dfInput = dfInput[column]
        else:
            # check if only one column remaining
            assert len(dfInput.columns) == 1,\
                f"Input file for {fileName} has more than one value column: {dfInput.columns.to_list()}"
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
            #
            logging.info(f"Missing index {missingIndex} detected in {fileName}. Input dataframe is extended by this index")
            _dfInputIndexTemp   = pd.MultiIndex.from_product([dfInput.index, requestedIndexValues],names=dfInput.index.names + [missingIndex])
            _dfInputIndexTemp   = _dfInputIndexTemp.reorder_levels(order=dfOutput.index.names)
            _dfInputTemp        = pd.Series(index=_dfInputIndexTemp, dtype=float)
            if column in dfInput.columns:
                dfInput         = _dfInputTemp.to_frame().apply(lambda row: dfInput.loc[
                    row.index.get_level_values(dfInput.index.names[0]), column].squeeze())
            else:
                dfInput         = _dfInputTemp.to_frame().apply(
                    lambda row: dfInput.loc[row.index.get_level_values(dfInput.index.names[0])].squeeze())
            if isinstance(dfInput,pd.DataFrame):
                dfInput = dfInput.squeeze()
            dfInput.index       = _dfInputTemp.index
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
