"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to read the data from input files, collect them into a dictionary.
==========================================================================================================================================================================="""

# IMPORT AND SETUP
import sys, logging, os
import pandas as pd
import numpy  as np

from deepdiff import DeepDiff


#%% CLASS DEFINITION AND METHODS
class Read:

    def carriers(self):
        """
        This method fills the initialised dictionary by reading the input carriers' data
        """
        logging.info('read data of all carriers')   
        
        for carrierSubset in self.analysis['subsets']['setCarriers']:
            
            for carrierName in self.data[carrierSubset].keys():           
                path      = self.paths[carrierSubset][carrierName]['folder']
                fileNames = [fileName for fileName in os.listdir(path) if (fileName.split('.')[-1]==self.analysis['fileFormat'])]

                for fileName in fileNames:                 
                    file      = pd.read_csv(path+fileName, header=0, index_col=None) 
                    dataInput = fileName.split('.')[0]
                    self.data[carrierSubset][carrierName][dataInput] = file
            

    def technologies(self):
        """
        This method fills the initialised dictionary by reading the technologies' data
        """
        logging.info('read data of all technologies')  
        
        for technologySubset in self.analysis['subsets']['setTechnologies']:
            
            for technologyName in self.data[technologySubset].keys():       
                path      = self.paths[technologySubset][technologyName]['folder']
                fileNames = [fileName for fileName in os.listdir(path) if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
                
                for fileName in fileNames:                                    
                    file      = pd.read_csv(path+fileName, header=0, index_col=None)                    
                    dataInput = fileName.split('.')[0]
                    self.data[technologySubset][technologyName][dataInput] = file 
              

    def nodes(self):
        """
        This method fills the initialised dictionary by reading the nodes' data
        """
        logging.info('read the nodes set') 
          
        path      = self.paths['setNodes']['folder']               
        nameNodes = self.analysis['dataInputs']['nameNodes']     
        fileNames = [fileName for fileName in os.listdir(path) if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
        
        for fileName in fileNames:                 
            file      = pd.read_csv(path+fileName, header=0, index_col=None)
            dataInput = fileName.split('.')[0] 
            self.data[dataInput] = file.loc[:, nameNodes]
            

    def times(self):
        """
        This method fills the initialised dictionary by reading the time horizon's data
        """
        logging.info('read the time steps set')    
        
        path          = self.paths['setTimeSteps']['folder'] 
        nameTimeSteps = self.analysis['dataInputs']['nameTimeSteps']
        fileNames     = [fileName for fileName in os.listdir(path) if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
        
        for fileName in fileNames:                   
            file      = pd.read_csv(path+fileName, header=0, index_col=None)
            dataInput = fileName.split('.')[0]    
            self.data[dataInput] = file.loc[:, nameTimeSteps]


    def scenarios(self):
        """
        This method fills the initialised dictionary by reading the scenarios' data
        """
        logging.info('read the scenarios set') 
         
        path          = self.paths['setScenarios']['folder']
        nameScenarios = self.analysis['dataInputs']['nameScenarios']
        fileNames     = [fileName for fileName in os.listdir(path) if (fileName.split('.')[-1]==self.analysis['fileFormat'])]
        
        for fileName in fileNames:             
            file      = pd.read_csv(path+fileName, header=0, index_col=None)
            dataInput = fileName.split('.')[0]
            self.data[dataInput] = file.loc[:, nameScenarios]
                                         