"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to create the paths to read the data from input files.
==========================================================================================================================================================================="""

import os

class Paths:
    
    def __init__(self):
        pass
    
    def data(self):
        
        # define path to access dataset related to the current analysis
        self.pathData = './/data//{}//'.format(self.analysis['spatialResolution'])    
        
        self.paths = dict()
        # create a dictionary with the keys based on the folders in pathData
        for folderName in next(os.walk(self.pathData))[1]:
            self.paths[folderName] = dict()
            self.paths[folderName]['folder'] = \
                self.pathData+'{}//'.format(folderName)
    
    def carriers(self):
        
        # add the paths for all the directories in each carrier subset    
        for carrierSubset in self.analysis['subsets']['setCarriers']:
            path = self.paths[carrierSubset]['folder']
            for carrier in next(os.walk(path))[1]:
                self.paths[carrierSubset][carrier] = dict()
                self.paths[carrierSubset][carrier]['folder'] = \
                    path+'{}//'.format(carrier)       
                
    def technologies(self):
                
        # add the paths for all the directories in technologies  
        for technologySubset in self.analysis['subsets']['setTechnologies']:        
            path = self.paths[technologySubset]['folder']
            for technology in next(os.walk(path))[1]:
                self.paths[technologySubset][technology] = dict()
                self.paths[technologySubset][technology]['folder'] = \
                    path+'{}//'.format(technology)
        
    
    