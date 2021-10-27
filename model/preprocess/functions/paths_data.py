# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import os

def data(self):
    
    # define path to access dataset related to the current analysis
    self.pathData = './/data//{}//'.format(self.analysis['case'])    
    
    self.paths = dict()
    # create a dictionary with the keys based on the folders from pathData
    for folderName in next(os.walk(self.pathData))[1]:
        self.paths[folderName] = dict()
        self.paths[folderName]['folder'] = \
            self.pathData+'{}//'.format(folderName)

def carriers(self):

    # add the paths for all the directories in carriers
    path = self.paths['carriers']['folder']
    for carrier in next(os.walk(path))[1]:
        self.paths['carriers'][carrier] = dict()
        self.paths['carriers'][carrier]['folder'] = \
            path+'{}//'.format(carrier)

def networks(self):   
         
    # add the paths for all the directories in networks
    path = self.paths['networks']['folder']
    for network in next(os.walk(path))[1]:
        self.paths['networks'][network] = dict()
        self.paths['networks'][network]['folder'] = \
            path+'{}//'.format(network)         
            
def technologies(self):
            
    # add the paths for all the directories in technologies  
    technologyTypes = ['production_technologies',\
                        'storage_technologies',\
                        'transport_technologies'
                        ]
    for technology_type in technologyTypes:        
        path = self.paths[technology_type]['folder']
        for technology in next(os.walk(path))[1]:
            self.paths[technology_type][technology] = dict()
            self.paths[technology_type][technology]['folder'] = \
                path+'{}//'.format(technology)      