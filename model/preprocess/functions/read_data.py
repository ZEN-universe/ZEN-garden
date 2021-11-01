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
import logging

def carriers(self):
    
    logging.info('read data of all the carriers')   
    
    carrierTypes = ['input_carriers',
                    'output_carriers'
                    ]
    
    for carrierType in carrierTypes:
        
        listNode = []
        listTime = []
        
        for carrierName in self.input[carrierType].keys():
            
            # Read the input data of the energy carrier
            for data_type in ['demand', 'supply', 'price']:
                
                path = self.paths[carrierType][carrierName]['folder']
                
                fileformat = 'csv'            
                filename = '{}.{}'.format(data_type, fileformat)
                        
                # table attributes                     
                file = pd.read_csv(path+filename, header=0, index_col=None)
                
                self.input[carrierType][carrierName][data_type] =\
                    file.loc[:,data_type].values     
                
                self.input[carrierType][carrierName]['node'] =\
                    file.loc[:,'node'].values
                    
                self.input[carrierType][carrierName]['time'] =\
                    file.loc[:,'time'].values  
                
                if list(file.loc[:,'node'].values) != []:
                    listNode.append(list(file.loc[:,'node'].values))
                if list(file.loc[:,'time'].values) != []:
                    listTime.append(list(file.loc[:,'time'].values))
        
        # verify that the input data are consinstent in time and nodes
        if (DeepDiff([listNode[0]]*len(listNode), listNode) != {}):
            # print(DeepDiff(listNode[0], listNode))
            logging.error('Inconsistent nodes in carrier input data')
    
        elif (DeepDiff([listTime[0]]*len(listTime), listTime) != {}):
            logging.error('Inconsistent time carrier input data')     
                
def networks(self):
    
    logging.info('read data of all the networks')       
    
    listMtxsize = []
    
    for networkName in self.input['networks'].keys():
        
        # Read the input data of the networks
        for data_type in ['distance']:
            
            path = self.paths['networks'][networkName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['networks'][networkName][data_type] =\
                file
                
            listMtxsize.append(list(file.shape))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
        # print(DeepDiff(listMtxsize[0], listMtxsize))
        logging.error('Inconsistent size in network input data')
        
def technologies(self):
    
    logging.info('read data of all the technologies')      
    
    listMtxsize = []
    
    for technologyName in self.input['production_technologies'].keys():
        
        # Read the input data of the energy carrier
        for data_type in ['attributes', 'availability_matrix']:
            
            path = self.paths['production_technologies'][technologyName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['production_technologies'][technologyName][data_type] =\
                file
            
            if data_type in ['availability_matrix']:
                listMtxsize.append(list(file.shape))
    
    # verify that the input data are consinstent in time and nodes
    if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
        # print(DeepDiff(listMtxsize[0], listMtxsize))
        logging.error('Inconsistent size in production technology'+\
                         ' availability matrix')   
    
    listMtxsize = [] 
    
    for technologyName in self.input['storage_technologies'].keys():
        
        # Read the input data of the energy carrier
        for data_type in ['attributes', 'availability_matrix',\
                          'max_capacity','min_capacity']:
            
            path = self.paths['storage_technologies'][technologyName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['storage_technologies'][technologyName][data_type] =\
                file   
                
            if data_type in ['availability_matrix','max_capacity',\
                             'min_capacity']:
                listMtxsize.append(list(file.shape))  

        # verify that the input data are consinstent in time and nodes
        if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
            error = 'Inconsistent size in storage technology'+\
                ' availability matrix'
            logging.error(error)                  

    listMtxsize = [] 

    for technologyName in self.input['transport_technologies'].keys():
        
        # Read the input data of the energy carrier
        for data_type in ['availability_matrix',\
                          'cost_per_distance', 'efficiency_per_distance']:
            
            path = self.paths['transport_technologies'][technologyName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.input['transport_technologies'][technologyName][data_type] =\
                file   
            
            listMtxsize.append(list(file.shape))  

        # verify that the input data are consinstent in time and nodes
        if (DeepDiff([listMtxsize[0]]*len(listMtxsize), listMtxsize) != {}):
        
            error = 'Inconsistent size in transport technology input data'
            logging.error(error)  
            
def nodes(self):

    path = self.paths['nodes']['folder']
    
    data_type = 'nodes'
    fileformat = 'csv'            
    filename = '{}.{}'.format(data_type, fileformat)  
    
    file = pd.read_csv(path+filename, header=0, index_col=False)
    
    indexNodes = np.arange(0, file['nodes'].values.size)
    
    for indexNode in indexNodes:
        self.input['nodes'][file['nodes'].iloc[indexNode]] = indexNode
        
def times(self):
    
    path = self.paths['times']['folder']
    
    data_type = 'times'
    fileformat = 'csv'            
    filename = '{}.{}'.format(data_type, fileformat)  
    
    file = pd.read_csv(path+filename, header=0, index_col=False)
    
    indexNodes = np.arange(0, file['times'].values.size)
    
    for indexNode in indexNodes:
        self.input['times'][file['times'].iloc[indexNode]] = indexNode  
        
def scenarios(self):
    
    path = self.paths['scenarios']['folder']
    
    data_type = 'scenarios'
    fileformat = 'csv'            
    filename = '{}.{}'.format(data_type, fileformat)  
    
    file = pd.read_csv(path+filename, header=0, index_col=False)
    
    indexNodes = np.arange(0, file['scenarios'].values.size)
    
    for indexNode in indexNodes:
        self.input['scenarios'][file['scenarios'].iloc[indexNode]] = indexNode         
        
                
                             
        
    