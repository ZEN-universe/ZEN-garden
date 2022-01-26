#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:34:37 2021

@author: Davide Tonelli, PhD candidate UCLouvain
         davide.tonelli@uclouvain.be
         davidetonelli@outlook.com
"""

import pandas as pd
import numpy as np
import os

def createFolder(path, folderName):
    """
    Method to create new folders if not already existing 
    """
    try:
        os.makedirs(path+folderName)
    except OSError:
        pass

class Create:
    
    def __init__(self, dictionary, analysis):
        
        self.pathMainFolder = '..//..//data//'
        self.dictionary = dictionary
        self.analysis = analysis

        # create main folder
        self.mainFolder = '..//data//'+self.dictionary['mainFolder']+'//'
        self.newFolder(self.mainFolder)

    def newFolder(self, folder):

        try:
            os.mkdir(folder)
        except OSError:
            pass

    def independentData(self, name, headerInSource):

        setName = 'set'+name
        folder = self.mainFolder+setName+'//'
        self.newFolder(folder)

        headerNames = self.analysis['headerDataInputs'][setName]

        data = pd.DataFrame()
        for header in headerNames:
            if header in self.dictionary:
                setattr(self, header, self.dictionary[header])
            elif header in headerInSource:
                setattr(self, header, self.dictionary['sourceData'].loc[:, headerInSource[header]].values)
            else:
                print(f'No input {setName} - 0 assigned')
                setattr(self, header, [0])
            data[header] = getattr(self, header)

        data.to_csv(folder+setName+'.csv', header=True, index=None)

    def carrierDependentData(self, name, headerInSource):

        setName = 'set'+name
        folder = self.mainFolder+setName+'//'
        self.newFolder(folder)

        scenarioHeader = self.analysis['headerDataInputs']['setScenarios'][0]
        nodeHeader = self.analysis['headerDataInputs']['setNodes'][0]
        timeHeader = self.analysis['headerDataInputs']['setTimeSteps'][0]

        headerNames = self.analysis['headerDataInputs'][setName]
        for header in headerNames:
            for carrier in headerInSource:
                # create carrier folder
                folder = self.mainFolder + setName + '//' + carrier + '//'
                self.newFolder(folder)
                # name in fixed input dictionary
                fixedInputKey = header+'_'+carrier
                if fixedInputKey in self.dictionary:
                    values = self.dictionary[fixedInputKey]
                elif header in headerInSource[carrier]:
                    values = self.dictionary['sourceData'].loc[:, headerInSource[carrier][header]]
                else:
                    print(f'No input {fixedInputKey} - 0 assigned')
                    values = [0]

                data = pd.DataFrame(columns=[scenarioHeader, timeHeader, nodeHeader, header])
                for scenario in getattr(self, self.analysis['headerDataInputs']['setScenarios'][0]):
                    for timeStep in getattr(self, self.analysis['headerDataInputs']['setTimeSteps'][0]):
                        for node in getattr(self, self.analysis['headerDataInputs']['setNodes'][0]):
                            if len(values) == 1:
                                valuesToAdd = {scenarioHeader: scenario, timeHeader: timeStep, nodeHeader:node,
                                               header:values[0]}
                            else:
                                valuesToAdd = {scenarioHeader: scenario, timeHeader: timeStep, nodeHeader:node,
                                               header:values[list(getattr(self, nodeHeader)).index(node)]}
                            data = data.append(valuesToAdd, ignore_index=True)

                data.to_csv(folder+header+'.csv', header=True, index=None)