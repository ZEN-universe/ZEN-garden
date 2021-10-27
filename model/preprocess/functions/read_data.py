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

def carriers(self):
    
    listNode = []
    listTime = []
    
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
                listNode.append(list(file.loc[:,'node'].values))
            if list(file.loc[:,'time'].values) != []:
                listTime.append(list(file.loc[:,'time'].values))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([listNode[0]]*len(listNode), listNode) != {}):
        print(DeepDiff(listNode[0], listNode))
        raise ValueError('Inconsistent nodes in carrier input data')

    elif (DeepDiff([listTime[0]]*len(listTime), listTime) != {}):
        raise ValueError('Inconsistent time carrier input data')
        
                
def networks(self):
    
    listMtxsize = []
    
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
                
            listMtxsize.append(list(file.shape))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
        print(DeepDiff(listMtxsize[0], listMtxsize))
        raise ValueError('Inconsistent size in network input data')
        
def technologies(self):
    
    listMtxsize = []
    
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
                listMtxsize.append(list(file.shape))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
        print(DeepDiff(listMtxsize[0], listMtxsize))
        raise ValueError('Inconsistent size in production technology'+\
                         ' availability matrix')   
    
    listMtxsize = [] 
    
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
                listMtxsize.append(list(file.shape))  

        # verify that the input data are consinstent in time and nodes
        if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
            error = 'Inconsistent size in storage technology'+\
                ' availability matrix'
            raise ValueError(error)                  

    listMtxsize = [] 

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
            
            listMtxsize.append(list(file.shape))  

        # verify that the input data are consinstent in time and nodes
        if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
        
            error = 'Inconsistent size in transport technology input data'
            raise ValueError(error)  
                             
        
    