# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import os

def Data(self):
    
    # define path to access dataset related to the current analysis
    self.path_data = './/data//{}//'.format(self.analysis['case'])    
    
    self.paths = dict()
    # create a dictionary with the keys based on the folders from path_data
    for folder_name in next(os.walk(self.path_data))[1]:
        self.paths[folder_name] = dict()
        self.paths[folder_name]['folder'] = \
            self.path_data+'{}//'.format(folder_name)

def Carriers(self):

    ## Carriers
    # add the paths for all the directories in carriers
    path = self.paths['carriers']['folder']
    for carrier in next(os.walk(path))[1]:
        self.paths['carriers'][carrier] = dict()
        self.paths['carriers'][carrier]['folder'] = \
            path+'{}//'.format(carrier)

def Networks(self):   
         
    ## Networks
    # add the paths for all the directories in networks
    path = self.paths['networks']['folder']
    for network in next(os.walk(path))[1]:
        self.paths['networks'][network] = dict()
        self.paths['networks'][network]['folder'] = \
            path+'{}//'.format(network)         
            
def Technologies(self):
            
    ## Technologies
    # add the paths for all the directories in technologies  
    technology_types = ['production_technologies',\
                        'storage_technologies',\
                        'transport_technologies'
                        ]
    for technology_type in technology_types:        
        path = self.paths[technology_type]['folder']
        for technology in next(os.walk(path))[1]:
            self.paths[technology_type][technology] = dict()
            self.paths[technology_type][technology]['folder'] = \
                path+'{}//'.format(technology)      