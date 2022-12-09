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

    def createPaths(self):
        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, technologies
        :return: dictionary all the paths for reading data
        """

        ## General Paths
        # define path to access dataset related to the current analysis
        self.pathData = self.analysis['dataset']
        assert os.path.exists(self.pathData), f"Folder for input data {self.analysis['dataset']} does not exist!"
        self.paths = dict()
        # create a dictionary with the keys based on the folders in pathData
        for folderName in next(os.walk(self.pathData))[1]:
            self.paths[folderName] = dict()
            self.paths[folderName]["folder"] = os.path.join(self.pathData, folderName)

        ## Carrier Paths
        # add the paths for all the directories in carrier folder
        path = self.paths["setCarriers"]["folder"]
        for carrier in next(os.walk(path))[1]:
            self.paths["setCarriers"][carrier] = dict()
            self.paths["setCarriers"][carrier]["folder"] = os.path.join(path, carrier)

        ## Technology Paths
        # add the paths for all the directories in technologies
        for technologySubset in self.analysis["subsets"]["setTechnologies"]:
            path = self.paths[technologySubset]["folder"]
            for technology in next(os.walk(path))[1]:
                self.paths[technologySubset][technology] = dict()
                self.paths[technologySubset][technology]["folder"] = os.path.join(path, technology)

    def check_existing_input_data(self):
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

    def check_existing_carrier_data(self, system):
        # check if carriers exist
        self.system = system
        for carrier in self.system["setCarriers"]:
            assert carrier in self.paths["setCarriers"].keys(), f"Carrier {carrier} selected in config does not exist in input data, excluded from model."

