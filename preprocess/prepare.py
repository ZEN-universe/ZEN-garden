"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to read the data from input files, collect them into a dictionary and convert the dictionary into a Pyomo
              compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""

import logging
from preprocess.functions.paths_data          import Paths
from preprocess.functions.initialise          import Init
from preprocess.functions.read_data           import Read
from preprocess.functions.modify_config       import UpdateConfig
from preprocess.functions.create_data         import Create
from preprocess.functions.fill_nlp_dictionary import FillNlpDict
from copy                                     import deepcopy

class Prepare:
    
    def __init__(self, config):
        """
        This class creates the dictionary containing all the input data
        organised per set according to the model formulation
        :param system: dictionary defining the system framework
        :return: dictionary containing all the input data
        """   
        # instantiate analysis object
        self.analysis = config.analysis
        
        # instantiate system object
        self.system = config.system

        # instantiate the solver object
        self.solver = config.solver

        # create a dictionary with the paths to access the model inputs
        self.createPaths()
        
        # only kept for NlpDict
        if self.solver["model"] == "MINLP":
            # initialise a dictionary with the keys of the data to be read
            self.initDict()

            # read data and store in the initialised dictionary
            self.readData()

            # update system and analysis with derived settings
            self.configUpdate()

            # collect data for nonlinear solver
            self.createNlpDict()
        
    def createPaths(self):
        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, technologies
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
        split by carriers, networks, technologies
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
        split by carriers, networks, technologies
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
        
    def createData(self):
        """
        This method creates data from the input dataset adding default values
        :return: new item in data dictionary
        """
        # create efficiency and avaialability matrices
        Create.conversionMatrices(self)

    def createNlpDict(self):
        """
        This method creates the dictionary of data passed to the nonlinear solver (nlpDict)
        :param system: dictionary defining the system framework
        :param data: dictionary containing all the input data
        :return: dictionary with data
        """
        self.nlpDict = {}

        # create input arrays based on solver configuration
        FillNlpDict.configSolver(self)

        # attach to the dictionary the interpolated functions
        FillNlpDict.functionNonlinearApproximation(self)

        # collect data concerning the variables' domain
        FillNlpDict.collectDomainExtremes(self)
    
    def checkExistingInputData(self):
        """ 
        This method checks the existing input data and only regards those elements for which folders exist.
        It is called in compile.py after the main Prepare routine.
        """
        system = deepcopy(self.system)

        # check if carriers exist
        for carrier in system["setCarriers"]:
            if carrier not in self.paths["setCarriers"].keys():
                logging.warning(f"Carrier {carrier} selected in config does not exist in input data, excluded from model.")
                system["setCarriers"].remove(carrier)

        # check if technologies exist
        system["setTechnologies"] = []
        for technologySubset in self.analysis["subsets"]["setTechnologies"]:
            for technology in system[technologySubset]:
                if technology not in self.paths[technologySubset].keys():
                    logging.warning(f"Technology {technology} selected in config does not exist in input data, excluded from model.")
                    system[technologySubset].remove(technology)
            system["setTechnologies"].extend(system[technologySubset])
        
        # return system
        return system