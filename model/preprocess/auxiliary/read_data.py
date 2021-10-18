# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import pandas as pd
import numpy as np

def Carriers(self):
    
    for carrier in self.input['Carriers'].keys():
        
        # Read properties of the energy carrier
        path = '{}{}//'.format(self.paths['Carriers'], carrier)
        
        file = pd.read_csv(\
            path+'properties.csv', header=0, index_col=0)
        
        for attribute in file.index:
            self.input['Carriers'][carrier][attribute] =\
                file.loc[attribute,'value']
        
        # Read properties of the energy carrier demand and supply
        for data_type in ['demand', 'supply']:
            
            self.input['Carriers'][carrier][data_type] = dict()
            
            path = '{}{}//{}//'.format(self.paths['Carriers'],
                carrier, data_type)
            
            # scalar attributes
            file = pd.read_csv(\
                path+'attributes.csv', header=0, index_col=0)
            
            for attribute in file.index:
                self.input['Carriers'][carrier][data_type][attribute] =\
                    file.loc[attribute,'value']
                    
            # table attributes                
            self.input['Carriers'][carrier][data_type]['table'] = pd.read_csv(
                path+'values.csv', header=0, index_col=None)
                
def Network(self):
    
    path = self.paths['Network']
    
    file = pd.read_csv(\
        path+'file.csv', header=0, index_col=None)
    
    # size of the network
    self.input['Network']['size'] = file.loc[:,'nodes'].size
    
    # create array with nodes idx 
    self.input['Network']['idx'] = np.arange(0, 
        self.input['Network']['size'], dtype=np.int)
    
    # create array with names of nodes ordered according to input file
    self.input['Network']['nodes'] = file.loc[:,'nodes'].values
    # create array with coordinates of the nodes
    self.input['Network']['X'] = file.loc[:,'X'].values
    self.input['Network']['Y'] = file.loc[:,'Y'].values

    # create a dictionary associating nodes' index to name   
    self.input['Network']['idx_to_name'] = dict()   
    # create a dictionary associating nodes' name to index   
    self.input['Network']['name_to_idx'] = dict()  
    
    for idx in self.input['Network']['idx']:
        
        name = self.input['Network']['nodes'][idx]
        
        self.input['Network']['idx_to_name'][idx] = name

        self.input['Network']['name_to_idx'][name] = idx

def Technologies(self):
    
    for technology in self.input['Technologies'].keys():
        
        path = '{}{}//'.format(
            self.paths['Technologies'],
            technology)
        
        file = pd.read_csv(\
            path+'attributes.csv', header=0, index_col=0)
        
        for attribute in file.index:
            self.input['Technologies'][technology][attribute] =\
                file.loc[attribute,'value']
        
        
    