"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to extract the input data from the provided input files
==========================================================================================================================================================================="""
import numpy as np
import pandas as pd
import os

class DataInput():
    def __init__(self,system,analysis,solver,energySystem = None):
        """ data input object to extract input data
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver 
        :param energySystem: instance of class <EnergySystem> to define energySystem """
        self.system         = system
        self.analysis       = analysis
        self.solver         = solver
        self.energySystem   = energySystem

    def extractInputData(self, folderPath,manualFileName=None,indexSets=[],column = None):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files 
        :param manualFileName: name of selected file. If only one file in folder, not used
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :return dataDict: dictionary with attribute values """
        # get system attribute
        fileFormat  = self.analysis["fileFormat"]
        indexNames  = self.analysis['dataInputs']
        system      = self.system
        # select data
        fileNames = [fileName.split('.')[0] for fileName in os.listdir(folderPath) if (fileName.split('.')[-1]==fileFormat)]
        assert (manualFileName in fileNames or len(fileNames) == 1), "Selection of files was ambiguous. Select folder with single input file or select specific file by name"
        for fileName in fileNames:
            if len(fileNames) > 1 and fileName != manualFileName:
                continue
            # table attributes                     
            dfInput = pd.read_csv(folderPath+fileName+'.'+fileFormat, header=0, index_col=None) 
            break 
        # select and drop scenario
        if indexNames["nameScenarios"] in dfInput.columns:
            dfInput = dfInput[dfInput[indexNames["nameScenarios"]]==system['setScenarios']].drop(indexNames["nameScenarios"],axis=1)
        # set index by indexSets
        assert set(indexSets).intersection(set(dfInput.columns)) == set(indexSets), f"requested index sets {set(indexSets) - set(indexSets).intersection(set(dfInput.columns))} are missing from input file for {fileName}"
        dfInput = dfInput.set_index(indexSets)
        # select rows
        indexList = []
        for index in indexSets:
            for indexName in indexNames:
                if index == indexNames[indexName]:
                    indexList.append(system[indexName.replace("name","set")])
        # create pd.MultiIndex and select data
        indexMultiIndex = pd.MultiIndex.from_product(indexList)
        if column:
            assert column in dfInput.columns, f"Requested column {column} not in columns {dfInput.columns.to_list()} of input file {fileName}"
            dfInput = dfInput[column]
        else:
            # check if only one column remaining
            assert len(dfInput.columns) == 1, f"Input file for {fileName} has more than one value column: {dfInput.columns.to_list()}"
            dfInput = dfInput.squeeze(axis=1)
        dfInput = dfInput.loc[indexMultiIndex]
        # convert to dict
        dataDict = dfInput.to_dict()
        return dataDict
    
    def extractTransportInputData(self, folderPath,manualFileName=None,indexSets=[]):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files 
        :param manualFileName: name of selected file. If only one file in folder, not used
        :param indexSets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :return dataDict: dictionary with attribute values  """
        # get system attribute
        fileFormat  = self.analysis["fileFormat"]
        indexNames  = self.analysis['dataInputs']
        system      = self.system
        
        # select data
        fileNames = [fileName.split('.')[0] for fileName in os.listdir(folderPath) if (fileName.split('.')[-1]==fileFormat)]
        assert (manualFileName in fileNames or len(fileNames) == 1), "Selection of files was ambiguous. Select folder with single input file or select specific file by name"
        for fileName in fileNames:
            if len(fileNames) > 1 and fileName != manualFileName:
                continue
            # table attributes                     
            dfInput = pd.read_csv(folderPath+fileName+'.'+fileFormat, header=0, index_col=None).set_index("node")
            break 
        # select indizes
        if indexSets:
            indexList = [self.energySystem.setEdges]
            for index in indexSets:
                for indexName in indexNames:
                    if index == indexNames[indexName]:
                        indexList.append(system[indexName.replace("name","set")])
            indexMultiIndex = pd.MultiIndex.from_product(indexList)
        else:
            indexMultiIndex = self.energySystem.setEdges
        # fill dict 
        dataDict = {}
        for index in indexMultiIndex:
            if isinstance(index,tuple):
                _node,_nodeAlias = self.energySystem.setNodesOnEdges[index[0]]
            else:
                _node,_nodeAlias = self.energySystem.setNodesOnEdges[index]
            dataDict[index] = dfInput.loc[_node,_nodeAlias]
        
        return dataDict

    def extractAttributeData(self, folderPath,attributeName):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files 
        :param attributeName: name of selected attribute
        :return attributeValue: attribute value """
        # select data
        fileName = "attributes.csv"
        assert fileName in os.listdir(folderPath), f"Folder {folderPath} does not contain '{fileName}'"       
        dfInput = pd.read_csv(folderPath+fileName, header=0, index_col=None).set_index("index").squeeze(axis=1)
        # set index by indexSets
        assert attributeName in dfInput.index, f"Attribute '{attributeName}' not in {fileName} in {folderPath}"
        attributeValue = dfInput.loc[attributeName]
        try:
            return float(attributeValue)
        except:
            return attributeValue
    
    def extractConversionCarriers(self, folderPath):
        """ reads input data and extracts conversion carriers
        :param folderPath: path to input files 
        :return carrierDict: dictionary with input and output carriers of technology """
        carrierDict = {}
        # get carriers
        for carrier in ["inputCarrier","outputCarrier"]:
            _carrierString = self.extractAttributeData(folderPath,carrier)
            _carrierList = _carrierString.strip().split(" ")
            for _carrierItem in _carrierList:
                # check if carrier in carriers of model
                assert _carrierItem in self.system["setCarriers"], f"{carrier} '{_carrierItem}' is not in carriers of model ({self.system['setCarriers']})"
            carrierDict[carrier] = _carrierList

        return carrierDict

    def extractPWAData(self, folderPath):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files 
        :return PWADict: dictionary with PWA parameters """
        # get system attribute
        fileFormat = self.analysis["fileFormat"]
        # select data
        PWADict = {}
        for type in self.analysis["nonlinearTechnologyApproximation"]:
            # extract all data values
            PWADict[type] = {}
            nonlinearValues = {}
            assert f"nonlinear{type}.{fileFormat}" in os.listdir(folderPath), f"File 'nonlinear{type}.{fileFormat}' does not exist in {folderPath}"
            dfInputNonlinear = pd.read_csv(folderPath+"nonlinear" + type + '.'+fileFormat, header=0, index_col=None)
            if type == "Capex":
                # make absolute capex
                dfInputNonlinear["capex"] = dfInputNonlinear["capex"]*dfInputNonlinear["capacity"]
            for column in dfInputNonlinear.columns:
                nonlinearValues[column] = dfInputNonlinear[column].to_list()
            # extract PWA breakpoints
            assert f"breakpointsPWA{type}.{fileFormat}" in os.listdir(folderPath), f"File 'breakpointsPWA{type}.{fileFormat}' does not exist in {folderPath}"
            dfInputBreakpoints = pd.read_csv(folderPath+"breakpointsPWA" + type + '.'+fileFormat, header=0, index_col=None)
            # assert that breakpoint variable (x variable in nonlinear input)
            assert dfInputBreakpoints.columns[0] in dfInputNonlinear.columns, f"breakpoint variable for PWA '{dfInputBreakpoints.columns[0]}' is not in nonlinear variables [{dfInputNonlinear.columns}]"
            breakpointVariable = dfInputBreakpoints.columns[0]
            breakpoints = dfInputBreakpoints[breakpointVariable].to_list()

            PWADict[type][breakpointVariable] = breakpoints
            for valueVariable in nonlinearValues:
                if valueVariable != breakpointVariable:
                    PWADict[type][valueVariable] = list(np.interp(breakpoints,nonlinearValues[breakpointVariable],nonlinearValues[valueVariable]))
        
        return PWADict

    @staticmethod
    def calculateEdgesFromNodes(setNodes):
        """ calculates setNodesOnEdges from setNodes
        :param setNodes: list of nodes in model 
        :return setNodesOnEdges: dict with edges and corresponding nodes """
        setNodesOnEdges = {}
        for node in setNodes:
            for nodeAlias in setNodes:
                if node != nodeAlias:
                    setNodesOnEdges[node+"-"+nodeAlias] = (node,nodeAlias)
        return setNodesOnEdges