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
        
        self.headerScenario = self.analysis['dataInputs']['nameScenarios']
        self.headerNodes = self.analysis['dataInputs']['nameNodes']  
        self.headerTimeSteps = self.analysis['dataInputs']['nameTimeSteps']
    
    def mainFolder(self, folderName):
        
        createFolder(self.pathMainFolder, folderName)
        
    def secondLevelFolders(self):
        
        for key in self.dictionary.keys():
            
            if key != 'mainFolder':
                
                path = '{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'])
                createFolder(path, key)
        
    def thirdLevelFolders(self,):
        
        for key in self.dictionary.keys():
            
            if key not in ['mainFolder', 'setNodes', 'setScenarios', 'setTimeSteps']:
                
                for item in self.dictionary[key]:
                    path = '{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], key)
                    createFolder(path, item)
        
    def carriersInFiles(self):
        
        for fileName in ['availabilityCarrier', 'exportPriceCarrier', 'importPriceCarrier']:
            ext = '.csv'
            columns = [self.headerScenario, self.headerTimeSteps, self.headerNodes, fileName]
            numberIndexes = len(self.dictionary['setScenarios'])*len(self.dictionary['setNodes']['Names'])*len(self.dictionary['setTimeSteps'])
            file = pd.DataFrame(columns=columns, index=np.arange(numberIndexes))
            
            idx = 0
            for scenarioName in self.dictionary['setScenarios']:
                for timeStepName in self.dictionary['setTimeSteps']:
                    for nodeName in self.dictionary['setNodes']['Names']:
                        
                        file.loc[idx, self.headerScenario] = scenarioName
                        file.loc[idx, self.headerTimeSteps] = timeStepName
                        file.loc[idx, self.headerNodes] = nodeName 
                        
                        idx +=1
            
            for carrierName in  self.dictionary['setInputCarriers']:
                path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setInputCarriers', carrierName)
                file.to_csv(path+fileName+ext, header=True, index=False)
                
            
    def carriersOutFiles(self):
        
        for fileName in ['demandCarrier', 'exportPriceCarrier', 'importPriceCarrier']:
            ext = '.csv'
            columns = [self.headerScenario, self.headerTimeSteps, self.headerNodes, fileName]
            file = pd.DataFrame(columns=columns)
            
            idx = 0
            for scenarioName in self.dictionary['setScenarios']:
                for timeStepName in self.dictionary['setTimeSteps']:
                    for nodeName in self.dictionary['setNodes']['Names']:
                        
                        file.loc[idx, self.headerScenario] = scenarioName
                        file.loc[idx, self.headerTimeSteps] = timeStepName
                        file.loc[idx, self.headerNodes] = nodeName 
                        
                        idx +=1
            
            for carrierName in  self.dictionary['setOutputCarriers']:
                
                path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setOutputCarriers', carrierName)
                file.to_csv(path+fileName+ext, header=True, index=False)
                
    def nodesFiles(self):
        
        columns = [self.headerNodes, 'x', 'y']
        fileName = 'setNodes'
        ext = '.csv'            
        file = pd.DataFrame(columns=columns)     
        
        idx = 0
        for nodeName in self.dictionary['setNodes']['Names']:
            
            file.loc[idx, self.headerNodes] = nodeName 
            file.loc[idx, 'x'] = self.dictionary['setNodes']['XCoord'][self.dictionary['setNodes']['Names'].index(nodeName)]
            file.loc[idx, 'y'] = self.dictionary['setNodes']['YCoord'][self.dictionary['setNodes']['Names'].index(nodeName)]
            idx +=1        
            
        path = '{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setNodes')
        file.to_csv(path+fileName+ext, header=True, index=False) 
        
    def scenariosFiles(self):
        
        columns = [self.headerScenario]
        fileName = 'setScenarios'
        ext = '.csv'            
        file = pd.DataFrame(columns=columns)     
        
        idx = 0
        for scenarioName in self.dictionary['setScenarios']:
            
            file.loc[idx, self.headerScenario] = scenarioName 
            idx +=1        
            
        path = '{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setScenarios')
        file.to_csv(path+fileName+ext, header=True, index=False) 
        
    def timeStepsFiles(self):
        
        columns = [self.headerTimeSteps]
        fileName = 'setTimeSteps'
        ext = '.csv'            
        file = pd.DataFrame(columns=columns)     
        
        idx = 0
        for timeStepName in self.dictionary['setTimeSteps']:
            
            file.loc[idx, self.headerTimeSteps] = timeStepName 
            idx +=1        
            
        path = '{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setTimeSteps')
        file.to_csv(path+fileName+ext, header=True, index=False)        

    def conversionFiles(self):

        for fileName in ['availabilityConversion']:
            ext = '.csv'
            columns = [self.headerScenario, self.headerTimeSteps, self.headerNodes, fileName]
            file = pd.DataFrame(columns=columns)
            
            idx = 0
            for scenarioName in self.dictionary['setScenarios']:
                for timeStepName in self.dictionary['setTimeSteps']:
                    for nodeName in self.dictionary['setNodes']['Names']:
                        
                        file.loc[idx, self.headerScenario] = scenarioName
                        file.loc[idx, self.headerTimeSteps] = timeStepName
                        file.loc[idx, self.headerNodes] = nodeName 
                        
                        idx +=1
            
            for carrierName in  self.dictionary['setConversionTechnologies']:
                
                path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setConversionTechnologies', carrierName)
                file.to_csv(path+fileName+ext, header=True, index=False)
        
        fileName = 'attributes'
        columns = ['attributes']
        indexes = ['minCapacityConversio', 'maxCapacityConversion']
        file = pd.DataFrame(columns=columns, index=indexes)
        file.index.name = 'index'
        
        for carrierName in self.dictionary['setConversionTechnologies']:
            
            path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setConversionTechnologies', carrierName)
            file.to_csv(path+fileName+ext, header=True, index=True)
            
    def storageFiles(self):
    
        for fileName in ['availabilityStorage']:
            ext = '.csv'
            columns = [self.headerScenario, self.headerTimeSteps, self.headerNodes, fileName]
            file = pd.DataFrame(columns=columns)
            
            idx = 0
            for scenarioName in self.dictionary['setScenarios']:
                for timeStepName in self.dictionary['setTimeSteps']:
                    for nodeName in self.dictionary['setNodes']['Names']:
                        
                        file.loc[idx, self.headerScenario] = scenarioName
                        file.loc[idx, self.headerTimeSteps] = timeStepName
                        file.loc[idx, self.headerNodes] = nodeName 
                        
                        idx +=1
            
            for storageName in self.dictionary['setStorageTechnologies']:
                
                path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setStorageTechnologies', storageName)
                file.to_csv(path+fileName+ext, header=True, index=False)
        
        for fileName in ['maxCapacityStorage', 'minCapacityStorage']:
            columns = [self.headerNodes, fileName]
            file = pd.DataFrame(columns=columns)
            file[self.headerNodes] = self.dictionary['setNodes']['Names']
            
            for storageName in self.dictionary['setStorageTechnologies']:
                path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setStorageTechnologies', storageName)
                file.to_csv(path+fileName+ext, header=True, index=False)     
            
    def transportFiles(self):
        
        indexes = self.dictionary['setNodes']['Names']
        columns = self.dictionary['setNodes']['Names']
        file = pd.DataFrame(columns=columns, index=indexes)
        file.index.name = 'node'
        ext = '.csv'
        
        for transportName in self.dictionary['setTransportTechnologies']:
            for fileName in ['availabilityTransport', 'costPerDistance', 'distance', 'efficiencyPerDistance']:
                path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setTransportTechnologies', transportName)
                file.to_csv(path+fileName+ext, header=True, index=True)  
                
            
                   