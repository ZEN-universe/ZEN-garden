"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to modify the config dictionary based on existing inputs from config and default_config
==========================================================================================================================================================================="""
import numpy as np


class UpdateConfig:

    def __init__(self):
        pass

    def createSetsFromSubsets(self):

        # create a new list per set name
        for setName in self.analysis['subsets'].keys():
            self.system[setName] = []

        # extend the list of elements in the set with all the items of the single subset
        for setName in self.analysis['subsets'].keys():
            for subsetName in self.analysis['subsets'][setName]:
                self.system[setName].extend(self.system[subsetName])

    def createSupportPoints(self):

        technologySubset = 'setConversionTechnologies'
        # types = self.analysis['linearTechnologyApproximation'].keys()
        for technologyName in self.system[technologySubset]:
            for type in types:
                if technologyName in self.analysis['nonlinearTechnologyApproximation'][type]:
                    pass
                else:
                    if technologyName in self.analysis['linearTechnologyApproximation'][type]:
                        df = self.data[technologySubset][technologyName][f'linear{type}']
                    else:
                        df = self.data[technologySubset][technologyName][f'PWA{type}']
                    setName = f'setSegments{type}{technologyName}'
                    self.system[setName] = df.index.values
