"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to read the data from input files, collect them into a dictionary and convert the dictionary into a Pyomo
                compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""
import os
import pandas as pd
from model.preprocess.functions.paths_data import Paths
from model.preprocess.functions.initialise import Init
from model.preprocess.functions.read_data import Read
from model.preprocess.functions.fill_pyomo_dictionary import FillPyoDict

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
        
        # convert data into a pyomo dictinary
        self.createPyoDict()

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
        
        
        
    def checkData(self):
        # TODO: define a routine to check the consistency of the data w.r.t.
        # the nodes, times and scenarios
        pass
        