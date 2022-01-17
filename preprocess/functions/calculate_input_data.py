"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to read and calculate the input data from the provided input files
==========================================================================================================================================================================="""

import pandas as pd
import os

class DataInput():
    def __init__(self,system,analysis,grid):
        """ data input object to calculate input data
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param grid: instance of class <Element> to define grid """
        self.system     = system
        self.analysis   = analysis
        self.grid       = grid

    def extractInputData(self, folderPath,manualFileName=None,indexSets=[]):
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
        # check if only one column remaining
        assert len(dfInput.columns) == 1, f"Input file for {fileName} has more than one value column: {dfInput.columns.to_list()}"
        # select rows
        indexList = []
        for index in indexSets:
            for indexName in indexNames:
                if index == indexNames[indexName]:
                    indexList.append(system[indexName.replace("name","set")])
        # create pd.MultiIndex and select data
        indexMultiIndex = pd.MultiIndex.from_product(indexList)
        dfInput = dfInput.loc[indexMultiIndex]
        # convert to dict
        dataDict = dfInput.squeeze(axis=1).to_dict()
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
            indexList = [self.grid.setEdges]
            for index in indexSets:
                for indexName in indexNames:
                    if index == indexNames[indexName]:
                        indexList.append(system[indexName.replace("name","set")])
            indexMultiIndex = pd.MultiIndex.from_product(indexList)
        else:
            indexMultiIndex = self.grid.setEdges
        # fill dict 
        dataDict = {}
        for index in indexMultiIndex:
            if isinstance(index,tuple):
                _node,_nodeAlias = self.grid.setNodesOnEdges[index[0]]
            else:
                _node,_nodeAlias = self.grid.setNodesOnEdges[index]
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
    
    
    def extractConversionCarriers(self, folderPath,referenceCarrier,manualFileName=None):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files 
        :param referenceCarrier: name of reference carrier of technology
        :param manualFileName: name of selected file. If only one file in folder, not used
        :return inputCarriers: input carriers of technology
        :return outputCarriers: output carriers of technology """
        # get system attribute
        fileFormat = self.analysis["fileFormat"]
        # select data
        fileNames = [fileName.split('.')[0] for fileName in os.listdir(folderPath) if (fileName.split('.')[-1]==fileFormat)]
        assert (manualFileName in fileNames or len(fileNames) == 1), "Selection of files was ambiguous. Select folder with single input file or select specific file by name"
        for fileName in fileNames:
            if len(fileNames) > 1 and fileName != manualFileName:
                continue
            # table attributes                     
            dfInput = pd.read_csv(folderPath+fileName+'.'+fileFormat, header=0, index_col=None).set_index("carrier")
            break 
        # select inputCarriers and outputCarriers
        # TODO: IMPORTANT: currently just referenceCarrier = outputCarriers, everything else = inputCarriers, because of error in model
        outputCarriers = referenceCarrier
        inputCarriers = list(set(dfInput.index)-set(referenceCarrier))
        
        return inputCarriers,outputCarriers

    def extractPWAData(self, folderPath,tech):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param folderPath: path to input files 
        :param tech: name of technology
        :return PWADict: dictionary with PWA parameters """
        # get system attribute
        fileFormat = self.analysis["fileFormat"]
        # select data
        PWADict = {}
        for type in self.analysis["nonlinearTechnologyApproximation"]:
            if tech not in self.analysis["nonlinearTechnologyApproximation"][type]:
                # model as PWA/linear
                assert f"PWA{type}.{fileFormat}" in os.listdir(folderPath), f"File 'PWA{type}.{fileFormat}' does not exist in {folderPath}"
                PWADict[type] = {}
                dfInput = pd.read_csv(folderPath+"PWA" + type + '.'+fileFormat, header=0, index_col=None)
                for PWAParameter in dfInput.columns:
                    PWADict[type][PWAParameter] = {}
                    for segment in dfInput.index:
                        PWADict[type][PWAParameter][segment] = dfInput.loc[segment,PWAParameter]
            else:
                # model as nonlinear
                raise NotImplementedError("the parameter extraction for the nonlinear technology approximation are not yet implemented.")
        
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