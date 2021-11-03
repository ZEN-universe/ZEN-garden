"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to read the data from input files, collect them into a dictionary.
==========================================================================================================================================================================="""

import pandas as pd
import numpy as np
from deepdiff import DeepDiff
import sys
import logging
import os

class Read:
    
    def __init__(self):
        pass

    def carriers(self):
        
        logging.info('read data of all the carriers')   
        
        for carrierSubset in self.analysis['carrierSubsets']:
            
            for carrierName in self.data[carrierSubset].keys():
                
                path = self.paths[carrierSubset][carrierName]['folder']
                
                fileNames = [fileName for fileName in os.listdir(path)\
                             if (fileName.split('.')[-1]==self.analysis['fileFormat'])]

                for fileName in fileNames:
                            
                    # table attributes                     
                    file = pd.read_csv(path+fileName, header=0, index_col=None) 
                    
                    dataInput = fileName.split('.')[0]
                    self.data[carrierSubset][carrierName][dataInput] = file
            
    def technologies(self):
        
        logging.info('read data of technologies')  
        
        for technologySubset in self.analysis['technologySubsets']:
            
            for technologyName in self.data[technologySubset].keys():
        
                path = self.paths[technologySubset][technologyName]['folder']
                
                fileNames = [fileName for fileName in os.listdir(path)\
                             if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
                
                for fileName in fileNames:                 
                            
                    # table attributes                     
                    file = pd.read_csv(path+fileName, header=0, index_col=None)                    
                    dataInput = fileName.split('.')[0]
    
                    self.data[technologySubset][technologyName][dataInput] = file 
              
    def nodes(self):
    
        logging.info('read the nodes set') 
          
        path = self.paths['setNodes']['folder']        
        
        nameNodes = self.analysis['dataInputs']['nameNodes']
        
        fileNames = [fileName for fileName in os.listdir(path)\
                     if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
        
        for fileName in fileNames:
                        
            file = pd.read_csv(path+fileName, header=0, index_col=None)
            dataInput = fileName.split('.')[0]
            
            self.data[dataInput] = file.loc[:, nameNodes]
            
    def times(self):
        
        logging.info('read the times set')    
        
        path = self.paths['setTimeSteps']['folder']
        
        nameTimeSteps = self.analysis['dataInputs']['nameTimeSteps']
        
        fileNames = [fileName for fileName in os.listdir(path)\
                     if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
        
        for fileName in fileNames:
                        
            file = pd.read_csv(path+fileName, header=0, index_col=None)
            dataInput = fileName.split('.')[0]
            
            self.data[dataInput] = file.loc[:, nameTimeSteps]
                    
    def scenarios(self):
        
        logging.info('read the scenarios set') 
         
        path = self.paths['setScenarios']['folder']
        
        nameScenarios = self.analysis['dataInputs']['nameScenarios']
        
        fileNames = [fileName for fileName in os.listdir(path)\
                     if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
        
        for fileName in fileNames:
                        
            file = pd.read_csv(path+fileName, header=0, index_col=None)
            dataInput = fileName.split('.')[0]
            
            self.data[dataInput] = file.loc[:, nameScenarios]
                                         
        
    