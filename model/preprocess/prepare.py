"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to read the data from input files, collect them into a dictionary and convert the dictionary into a 
              Pyomo-compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""

# IMPORT AND SETUP
import os
import pandas as pd

from model.preprocess.functions.paths_data            import Paths
from model.preprocess.functions.initialise            import Init
from model.preprocess.functions.read_data             import Read
from model.preprocess.functions.modify_config         import UpdateConfig
from model.preprocess.functions.create_data           import Create
from model.preprocess.functions.fill_pyomo_dictionary import FillPyoDict


#%% CLASS DEFINITION
class Prepare:
    
    def __init__(self, analysis, system):
        """
        This class creates the dictionary containing all input data, organised per set according to the model formulation.
        :param system: dictionary defining the system framework
        :return: dictionary containing all the input data
        """
        # instantiate the analysis and system properties
        self.analysis = analysis
        self.system   = system
        
        # create a dictionary with the paths to access the model inputs --> module Paths called from paths_data.py
        self.createPaths()
        
        # initialise a dictionary with the keys of the data to be read --> module Init called from initialise.py
        self.initDict() 
        
        # read data and store in the initialised dictionary --> module Read called called from read_data.py
        self.readData()    
        
        # update system and analysis with derived settings --> module UpdateConfig called from modify_config.py
        self.configUpdate()        
        
        # create new data items from default values and input data --> module Create called from create_data.py
        self.createData()
        
        # convert data into a pyomo dictinary --> module FillPyoDict called from fill_pyomo_dictionary.py
        self.createPyoDict()


#%% CLASS METHODS

    def createPaths(self):
        """
        This method creates a dictionary with the paths of the data split by carriers, networks, tecnhologies
        :return: dictionary all the paths for reading data
        """   
        # create paths of the data folders according to the input sets
        Paths.data(self)

        # create paths of the carriers' folders
        Paths.carriers(self)    

        # create paths of technologies' folders
        Paths.technologies(self)

    
    def initDict(self):
        """
        This method initialises a dictionary containing all the input data split by carriers, networks, tecnhologies
        :return: dictionary initialised with keys
        """        
        self.data = dict()
        
        # initialise the keys with the names of input carriers
        Init.carriers(self)

        # initialise the keys with the names of technologies         
        Init.technologies(self)

        # initialise the key of nodes
        Init.nodes(self)

        # initialise the key of times
        Init.times(self)  

        # initialise the key of scenarios
        Init.scenarios(self) 

        
    def readData(self):
        """
        This method fills in the dictionary with all the input data split by carriers, networks, tecnhologies
        :return: dictionary containing all the input data 
        """                       
        # fill the initialised dictionary by reading the input carriers' data        
        Read.carriers(self)     

        # fill the initialised dictionary by reading the technologies' data          
        Read.technologies(self)     

        # fill the initialised dictionary by reading the nodes' data         
        Read.nodes(self)

        # fill the initialised dictionary by reading the times' data           
        Read.times(self)    

        # fill the initialised dictionary by reading the scenarios' data       
        Read.scenarios(self) 


    def configUpdate(self):
        """
        This method creates new entries in the dictionaries of config
        :return: dictionaries in config with additional entries
        """  
        # create new list of sets from subsets
        UpdateConfig.createSetsFromSubsets(self)

        # create sets of support points for PWA
        UpdateConfig.createSupportPoints(self)        
                    
        
    def createData(self):
        """
        This method creates data from the input dataset adding default values
        :return: new item in data dictionary
        """
        # create efficiency and avaialability matrices
        Create.conversionMatrices(self)
        

    def createPyoDict(self):
        """
        This method reshapes the input data dictionary into a dictionary with format compatible with Pyomo
        :param system: dictionary defining the system framework
        :param data: dictionary containing all the input data
        :return: dictionary with data based on system in Pyomo format      
        """
        self.pyoDict = {None:{}}
        
        # fill the dictionary with the sets based on system 
        FillPyoDict.sets(self)

        # fill the dictionary with the parameters related to the carrier
        FillPyoDict.carrierParameters(self)

        # fill the dictionary with the parameters related to the transport technology
        FillPyoDict.technologyTranspParameters(self)

        # fill the dictionary with the parameters related to the production and storage technology
        FillPyoDict.technologyProductionStorageParameters(self)

        # fill the dictionary with the parameters attributes of a technology
        FillPyoDict.attributes(self)

        # fill the dictionary with the conversion coefficients of a technology
        FillPyoDict.conversionBalanceParameters(self)

        # fill the dictionary with the PWA input data
        FillPyoDict.dataPWAApproximation(self)
        

    def checkData(self):
        # TODO: define a routine to check the consistency of the data w.r.t.
        # the nodes, times and scenarios
        pass
        