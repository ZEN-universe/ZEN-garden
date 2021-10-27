# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import pandas as pd
import numpy as np
from deepdiff import DeepDiff
import sys

def Carriers(self):
    
    node_ls = []
    time_ls = []
    
    for carrier in self.input['carriers'].keys():
        
        # Read the input data of the energy carrier
        for data_type in ['demand', 'supply', 'price']:
            
            path = self.paths['carriers'][carrier]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=None)
            
            self.input['carriers'][carrier][data_type] =\
                file.loc[:,data_type].values     
            
            self.input['carriers'][carrier]['node'] =\
                file.loc[:,'node'].values
                
            self.input['carriers'][carrier]['time'] =\
                file.loc[:,'time'].values  
            
            if list(file.loc[:,'node'].values) != []:
                node_ls.append(list(file.loc[:,'node'].values))
            if list(file.loc[:,'time'].values) != []:
                time_ls.append(list(file.loc[:,'time'].values))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([node_ls[0]]*len(node_ls), node_ls) != {}):
        print(DeepDiff(node_ls[0], node_ls))
        raise ValueError('Inconsistent nodes in carrier input data')

    elif (DeepDiff([time_ls[0]]*len(time_ls), time_ls) != {}):
        raise ValueError('Inconsistent time carrier input data')
        
                
def Networks(self):
    
    mtxsize_ls = []
    
    for network in self.input['networks'].keys():
        
        # Read the input data of the networks
        for data_type in ['distance']:
            
            path = self.paths['networks'][network]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['networks'][network][data_type] =\
                file
                
            mtxsize_ls.append(list(file.shape))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([mtxsize_ls[0]]*len(mtxsize_ls), mtxsize_ls) != {}):
        print(DeepDiff(mtxsize_ls[0], mtxsize_ls))
        raise ValueError('Inconsistent size in network input data')
        
def Technologies(self):
    
    mtxsize_ls = []
    
    for technology in self.input['production_technologies'].keys():
        
        # Read the input data of the energy carrier
        for data_type in ['attributes', 'availability_matrix']:
            
            path = self.paths['production_technologies'][technology]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['production_technologies'][technology][data_type] =\
                file
            
            if data_type in ['availability_matrix']:
                mtxsize_ls.append(list(file.shape))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([mtxsize_ls[0]]*len(mtxsize_ls), mtxsize_ls) != {}):
        print(DeepDiff(mtxsize_ls[0], mtxsize_ls))
        raise ValueError('Inconsistent size in production technology'+\
                         ' availability matrix')   
    
    mtxsize_ls = [] 
    
    for technology in self.input['storage_technologies'].keys():
        
        # Read the input data of the energy carrier
        for data_type in ['attributes', 'availability_matrix',\
                          'max_capacity','min_capacity']:
            
            path = self.paths['storage_technologies'][technology]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['storage_technologies'][technology][data_type] =\
                file   
                
            if data_type in ['availability_matrix','max_capacity',\
                             'min_capacity']:
                mtxsize_ls.append(list(file.shape))  

        # verify that the input data are consinstent in time and nodes
        if (DeepDiff([mtxsize_ls[0]]*len(mtxsize_ls), mtxsize_ls) != {}):
            error = 'Inconsistent size in storage technology'+\
                ' availability matrix'
            raise ValueError(error)                  

    mtxsize_ls = [] 

    for technology in self.input['transport_technologies'].keys():
        
        # Read the input data of the energy carrier
        for data_type in ['availability_matrix',\
                          'cost_per_distance', 'efficiency_per_distance']:
            
            path = self.paths['transport_technologies'][technology]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['transport_technologies'][technology][data_type] =\
                file   
            
            mtxsize_ls.append(list(file.shape))  

        # verify that the input data are consinstent in time and nodes
        if (DeepDiff([mtxsize_ls[0]]*len(mtxsize_ls), mtxsize_ls) != {}):
        
            error = 'Inconsistent size in transport technology input data'
            raise ValueError(error)  
                             
        
    