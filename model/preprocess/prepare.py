# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================
#                                    PREPARATION: DEFINE PATHS AND REARRANGE DATA
# ======================================================================================================================

import os
import pandas as pd
import model.preprocess.functions.paths_data as paths
import model.preprocess.functions.initialise as init
import model.preprocess.functions.read_data as read

class Prepare:
    
    def __init__(self, analysis, system):
        """
        This class creates the dictionary containing all the input data
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data
        """
        
        # instantiate analysis object
        self.analysis = analysis
        
        # create a dictionary with the paths to access the model inputs
        self.createPaths()
        
        # initialise a dictionary with the keys of the data to be read
        self.initDict() 
        
        # read data and store in the initialised dictionary
        self.readData()

    def createPaths(self):
        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, tecnhologies
        :param analysis: dictionary defining the analysis framework
        :return: dictionary all the paths for reading data
        """
        
        # create paths of data folders: carriers, networks, technologies
        paths.Data(self)
        # create paths of carriers' folders
        paths.Carriers(self)
        # create paths of netwoks' folders        
        paths.Networks(self)
        # create paths of technologies' folders   
        paths.Technologies(self)        
        
    def initDict(self):
        """
        This method initialises a dictionary containing all the input data
        split by carriers, networks, tecnhologies
        :return: dictionary initialised with keys
        """        
        
        self.input = dict()
        
        # initialise the keys with the carriers' name
        init.Carriers(self)
        # initialise the keys with the networks' name      
        init.Networks(self)
        # initialise the keys with the technologies' name           
        init.Technologies(self)
        
    def readData(self):
        """
        This method fills in the dictionary with all the input data
        split by carriers, networks, tecnhologies
        :return: dictionary containing all the input data 
        """                
        
        # fill the initialised dictionary by reading the carriers' data
        read.Carriers(self)
        # fill the initialised dictionary by reading the netwroks' data        
        read.Networks(self)
        # fill the initialised dictionary by reading the technologies' data          
        read.Technologies(self)
        
    
    
    
        