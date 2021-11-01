# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import os
def carriers(self):
    
    carrierTypes = ['input_carriers',
                    'output_carriers'
                    ]
    for carrierType in carrierTypes:      
        self.input[carrierType] = dict()
        path = self.paths[carrierType]['folder']
        
        # read all the folders in the carriers directory
        for carrierName in next(os.walk(path))[1]:
            self.input[carrierType][carrierName] = dict()
        
def networks(self):

    self.input['networks'] = dict() 
    path = self.paths['networks']['folder']
    
    # read all the folders in the networks directory
    for networkName in next(os.walk(path))[1]:
        self.input['networks'][networkName] = dict()
        
def technologies(self):
    
    technologyTypes = ['production_technologies',\
                       'storage_technologies',\
                       'transport_technologies'
                       ]
    
    for technologyType in technologyTypes:
        
        self.input[technologyType] = dict()
        path = self.paths[technologyType]['folder']
        
        # read all the folders in the directory of the specific type of 
        # technology
        for technologyName in next(os.walk(path))[1]:
            self.input[technologyType][technologyName] = dict()         
        
def nodes(self):
    
    self.input['nodes'] = dict()

def times(self):
    
    self.input['times'] = dict()   
    
def scenarios(self):
    
    self.input['scenarios'] = dict()     
    
    