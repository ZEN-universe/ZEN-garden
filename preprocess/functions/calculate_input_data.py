"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to read and calculate the input data from the provided input files
==========================================================================================================================================================================="""

import pandas as pd
import os

def readInputFiles(folderPath,fileFormat,manualFileName=None):
    """ reads input files and returns file 
    :param folderPath: path to input files 
    :param fileFormat: format of files
    :param manualFileName: name of selected file. If only one file in folder, not used
    :return dfInput: pd.Dataframe from input """
    fileNames = [fileName.split('.')[0] for fileName in os.listdir(folderPath) if (fileName.split('.')[-1]==fileFormat)]
    assert (manualFileName or len(fileNames) == 1), "Selection of files was ambiguous. Select folder with single input file or select specific file by name"
    for fileName in fileNames:
        if len(fileNames) > 1 and fileName != manualFileName:
            continue
        # table attributes                     
        dfInput = pd.read_csv(folderPath+fileName+'.'+fileFormat, header=0, index_col=None) 
    return dfInput

def extractInputData(folderPath,fileFormat,manualFileName=None,indexSets=[],scenario = None):
    """ reads input data and restructures the dataframe to return (multi)indexed dict
    :param folderPath: path to input files 
    :param fileFormat: format of files
    :param manualFileName: name of selected file. If only one file in folder, not used
    :param indexSets: index sets of attribute. Creates (multi)index 
    :
    :return dataDict: dictionary with attribute values """
    fileNames = [fileName.split('.')[0] for fileName in os.listdir(folderPath) if (fileName.split('.')[-1]==fileFormat)]
    assert (manualFileName in fileNames or len(fileNames) == 1), "Selection of files was ambiguous. Select folder with single input file or select specific file by name"
    for fileName in fileNames:
        if len(fileNames) > 1 and fileName != manualFileName:
            continue
        # table attributes                     
        dfInput = pd.read_csv(folderPath+fileName+'.'+fileFormat, header=0, index_col=None) 
    a=1

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