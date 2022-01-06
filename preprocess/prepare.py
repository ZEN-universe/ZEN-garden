"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to read the data from input files, collect them into a dictionary and convert the dictionary into a Pyomo
                compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""
import os
import pandas as pd
from preprocess.functions.modify_config import UpdateConfig
from preprocess.functions.paths_data import Paths
from preprocess.functions.initialise import Init
from preprocess.functions.read_data import Read
from preprocess.functions.create_data import Create
from preprocess.functions.fill_pyomo_dictionary import FillPyoDict
from preprocess.functions.fill_nlp_dictionary import FillNlpDict

class Prepare:
    
    def __init__(self, analysis, system):
        """
        This class creates the dictionary containing all the input data
        organised per set according to the model formulation
        :param system: dictionary defining the system framework
        :return: dictionary containing all the input data
        """
        
        # instantiate analysis object
        self.analysis = analysis
        
        # instantiate system object
        self.system = system
        
        # create a dictionary with the paths to access the model inputs
        self.createPaths()
        
        # initialise a dictionary with the keys of the data to be read
        self.initDict() 
        
        # read data and store in the initialised dictionary
        self.readData()
        
        # update system and analysis with derived settings
        self.configUpdate()        
        
        # create new data items from default values and input data
        self.createData()
        
        # convert data into a pyomo dictionary
        self.createPyoDict()

    def configUpdate(self):
        """
        This method creates new entries in the dictionaries of config
        :return: dictionaries in config with additional entries
        """
        
        # create new list of sets from subsets
        UpdateConfig.createSetsFromSubsets(self)
        # create sets of support points for PWA
        UpdateConfig.createSupportPoints(self)
        
    def createPaths(self):
        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, tecnhologies
        :return: dictionary all the paths for reading data
        """
        
        # create paths of the data folders according to the input sets
        Paths.data(self)
        # create paths of the input carriers' folders
        Paths.carriers(self)    
        # create paths of technologies' folders   
        Paths.technologies(self)
        
    def initDict(self):
        """
        This method initialises a dictionary containing all the input data
        split by carriers, networks, tecnhologies
        :return: dictionary initialised with keys
        """        
        
        self.data = dict()
        
        # initialise the keys with the input carriers' name
        Init.carriers(self)
        # initialise the keys with the technologies' name           
        Init.technologies(self)
        # initialise the key of nodes
        Init.nodes(self)
        # initialise the key of times
        Init.times(self)  
        # initialise the key of scenarios
        Init.scenarios(self)             
        
    def readData(self):
        """
        This method fills in the dictionary with all the input data
        split by carriers, networks, tecnhologies
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
        
    def createData(self):
        """
        This method creates data from the input dataset adding default values
        :return: new item in data dictionary
        """
        
        # create efficiency and avaialability matrices
        Create.conversionMatrices(self)
        
    def createPyoDict(self):
        """
        This method reshapes the input data dictionary into a dictionary 
        with format compatible with Pyomo
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
        # fill the dictionary with the parameters related to the storage and the conversion technology
        FillPyoDict.technologyConversionStorageParameters(self)
        # fill the dictionary with the parameters attributes of a technology
        FillPyoDict.attributes(self)
        # fill the dictionary with the conversion coefficients of a technology
        FillPyoDict.conversionBalanceParameters(self)
        # fill the dictionary with the PWA input data
        FillPyoDict.dataPWAApproximation(self)

        self.nlpDict = {None:{}}
        
        # attach to the dictionary the interpolated functions
        FillNlpDict.functionNonlinearApproximation(self)
        
    def checkData(self):
        # TODO: define a routine to check the consistency of the data w.r.t.
        # the nodes, times and scenarios
        pass
        