"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to create the paths to read the data from input files.
==========================================================================================================================================================================="""

import os
import logging

class Paths:
    
    def __init__(self):
        pass
    
    def data(self):
        
        # define path to access dataset related to the current analysis
        self.pathData = './/data//{}//'.format(self.analysis['dataset'])    
        assert os.path.exists(self.pathData),f"Folder for input data {self.analysis['dataset']} does not exist!"
        self.paths = dict()
        # create a dictionary with the keys based on the folders in pathData
        for folderName in next(os.walk(self.pathData))[1]:
            self.paths[folderName] = dict()
            self.paths[folderName]['folder'] = \
                self.pathData+'{}//'.format(folderName)
    
    def carriers(self):
        
        # add the paths for all the directories in carrier folder
        path = self.paths['setCarriers']['folder']
        for carrier in next(os.walk(path))[1]:
            self.paths['setCarriers'][carrier] = dict()
            self.paths['setCarriers'][carrier]['folder'] = \
                path+'{}//'.format(carrier)       
                
    def technologies(self):
                
        # add the paths for all the directories in technologies  
        for technologySubset in self.analysis['subsets']['setTechnologies']:        
            path = self.paths[technologySubset]['folder']
            for technology in next(os.walk(path))[1]:
                self.paths[technologySubset][technology] = dict()
                self.paths[technologySubset][technology]['folder'] = \
                    path+'{}//'.format(technology)
        
    
    