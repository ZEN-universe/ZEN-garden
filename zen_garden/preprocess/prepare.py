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
        self.create_paths()

    def create_paths(self):
        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, technologies
        :return: dictionary all the paths for reading data
        """

        ## General Paths
        # define path to access dataset related to the current analysis
        self.path_data = self.analysis['dataset']
        assert os.path.exists(self.path_data), f"Folder for input data {self.analysis['dataset']} does not exist!"
        self.paths = dict()
        # create a dictionary with the keys based on the folders in path_data
        for folder_name in next(os.walk(self.path_data))[1]:
            self.paths[folder_name] = dict()
            self.paths[folder_name]["folder"] = os.path.join(self.path_data, folder_name)

        ## Carrier Paths
        # add the paths for all the directories in carrier folder
        path = self.paths["set_carriers"]["folder"]
        for carrier in next(os.walk(path))[1]:
            self.paths["set_carriers"][carrier] = dict()
            self.paths["set_carriers"][carrier]["folder"] = os.path.join(path, carrier)

        ## Technology Paths
        # add the paths for all the directories in technologies
        for technology_subset in self.analysis["subsets"]["set_technologies"]:
            path = self.paths[technology_subset]["folder"]
            for technology in next(os.walk(path))[1]:
                self.paths[technology_subset][technology] = dict()
                self.paths[technology_subset][technology]["folder"] = os.path.join(path, technology)

    def check_existing_input_data(self):
        """This method checks the existing input data and only regards those elements for which folders exist.
        It is called in compile.py after the main Prepare routine."""

        # check if technologies exist
        self.system["set_technologies"] = []
        for technology_subset in self.analysis["subsets"]["set_technologies"]:
            for technology in self.system[technology_subset]:
                if technology not in self.paths[technology_subset].keys():
                    logging.warning(f"Technology {technology} selected in config does not exist in input data, excluded from model.")
                    self.system[technology_subset].remove(technology)
            self.system["set_technologies"].extend(self.system[technology_subset])
            # check subsets of technology_subset
            if technology_subset in self.analysis["subsets"].keys():
                for subset in self.analysis["subsets"][technology_subset]:
                    for technology in self.system[subset]:
                        if technology not in self.paths[technology_subset].keys():
                            logging.warning(f"Technology {technology} selected in config does not exist in input data, excluded from model.")
                            self.system[subset].remove(technology)
                    self.system[technology_subset].extend(self.system[subset])
                    self.system["set_technologies"].extend(self.system[subset])

    def check_existing_carrier_data(self, system):
        # check if carriers exist
        self.system = system
        for carrier in self.system["set_carriers"]:
            assert carrier in self.paths["set_carriers"].keys(), f"Carrier {carrier} selected in config does not exist in input data, excluded from model."
