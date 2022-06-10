"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class to read the data from input files, collect them into a dictionary and convert the dictionary into a Pyomo
              compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""

import logging
import os
from preprocess.functions.MINLP.initialise          import Init
from preprocess.functions.MINLP.read_data           import Read
from preprocess.functions.MINLP.fill_nlp_dictionary import FillNlpDict


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
        self.system   = config.system

        # instantiate the solver object
        self.solver   = config.solver

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

        ## General Paths
        # define path to access dataset related to the current analysis
        self.pathData = f".//data//{self.analysis['dataset']}//"
        assert os.path.exists(self.pathData), f"Folder for input data {self.analysis['dataset']} does not exist!"
        self.paths = dict()
        # create a dictionary with the keys based on the folders in pathData
        for folderName in next(os.walk(self.pathData))[1]:
            self.paths[folderName] = dict()
            self.paths[folderName]["folder"] = \
                self.pathData + f"{folderName}//"

        ## Carrier Paths
        # add the paths for all the directories in carrier folder
        path = self.paths["setCarriers"]["folder"]
        for carrier in next(os.walk(path))[1]:
            self.paths["setCarriers"][carrier] = dict()
            self.paths["setCarriers"][carrier]["folder"] = \
                path + f"{carrier}//"

        ## Technology Paths
        # add the paths for all the directories in technologies
        for technologySubset in self.analysis["subsets"]["setTechnologies"]:
            path = self.paths[technologySubset]["folder"]
            for technology in next(os.walk(path))[1]:
                self.paths[technologySubset][technology] = dict()
                self.paths[technologySubset][technology]["folder"] = \
                    path + f"{technology}//"
            # # add path for subsets of technologySubset
            # if technologySubset in self.analysis["subsets"].keys():
            #     for subset in self.analysis["subsets"][technologySubset]:
            #         path = self.paths[subset]["folder"]
            #         for technology in next(os.walk(path))[1]:
            #             self.paths[subset][technology] = dict()
            #             self.paths[subset][technology]["folder"] = \
            #                 path + f"{technology}//"

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
        # create a new list per set name
        for setName in self.analysis['subsets'].keys():
            self.system[setName] = []

        # extend the list of elements in the set with all the items of the single subset
        for setName in self.analysis['subsets'].keys():
            for subsetName in self.analysis['subsets'][setName]:
                self.system[setName].extend(self.system[subsetName])

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
        """This method checks the existing input data and only regards those elements for which folders exist.
        It is called in compile.py after the main Prepare routine."""

        # check if technologies exist
        self.system["setTechnologies"] = []
        for technologySubset in self.analysis["subsets"]["setTechnologies"]:
            for technology in self.system[technologySubset]:
                if technology not in self.paths[technologySubset].keys():
                    logging.warning(f"Technology {technology} selected in config does not exist in input data, excluded from model.")
                    self.system[technologySubset].remove(technology)
            self.system["setTechnologies"].extend(self.system[technologySubset])
            # check subsets of technologySubset
            if technologySubset in self.analysis["subsets"].keys():
                for subset in self.analysis["subsets"][technologySubset]:
                    for technology in self.system[subset]:
                        if technology not in self.paths[technologySubset].keys():
                            logging.warning(f"Technology {technology} selected in config does not exist in input data, excluded from model.")
                            self.system[subset].remove(technology)
                    self.system[technologySubset].extend(self.system[subset])
                    self.system["setTechnologies"].extend(self.system[subset])


    def checkExistingCarrierData(self, system):
        # check if carriers exist
        self.system = system
        for carrier in self.system["setCarriers"]:
            assert carrier in self.paths["setCarriers"].keys(), f"Carrier {carrier} selected in config does not exist in input data, excluded from model."

