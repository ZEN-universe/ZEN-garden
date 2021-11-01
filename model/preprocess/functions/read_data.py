"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description: Colleciton of methods used in the class Prepare
==========================================================================================================================================================================="""


import pandas as pd
import numpy as np
import sys
import logging

def carriers(self):
    
    logging.info('read data of all the carriers')   
    
    carrierTypes = ['input_carriers',
                    'output_carriers'
                    ]
    
    for carrierType in carrierTypes:

        for carrierName in self.data[carrierType].keys():
            
            # Read the input data of the energy carrier
            for dataType in ['demand', 'availability', 'importPrice', 'exportPrice']:
                
                path = self.paths[carrierType][carrierName]['folder']
                
                fileformat = 'csv'            
                filename = '{}.{}'.format(dataType, fileformat)
                        
                # table attributes                     
                file = pd.read_csv(path+filename,\
                                   header=0, index_col=None\
                                   ).set_index(['node', 'time', 'scenario'])
                
                self.data[carrierType][carrierName][dataType] =\
                    file
                
def networks(self):
    
    logging.info('read data of all the networks')       
    
    for networkName in self.data['networks'].keys():
        
        # Read the input data of the networks
        for data_type in ['distance']:
            
            path = self.paths['networks'][networkName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(data_type, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.data['networks'][networkName][data_type] =\
                file
        
def technologies(self):
    
    logging.info('read data of all the technologies')      
    
    for technologyName in self.data['production_technologies'].keys():
        
        # Read the input data of the energy carrier
        for dataType in ['attributes', 'availability_matrix']:
            
            path = self.paths['production_technologies'][technologyName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(dataType, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.data['production_technologies'][technologyName][dataType] =\
                file
    
    for technologyName in self.data['storage_technologies'].keys():
        
        # Read the input data of the energy carrier
        for dataType in ['attributes', 'availability_matrix',\
                          'max_capacity','min_capacity']:
            
            path = self.paths['storage_technologies'][technologyName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(dataType, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.data['storage_technologies'][technologyName][dataType] =\
                file   

    for technologyName in self.data['transport_technologies'].keys():
        
        # Read the input data of the energy carrier
        for dataType in ['availability_matrix',\
                          'cost_per_distance', 'efficiency_per_distance']:
            
            path = self.paths['transport_technologies'][technologyName]['folder']
            
            fileformat = 'csv'            
            filename = '{}.{}'.format(dataType, fileformat)
                    
            # table attributes                     
            file = pd.read_csv(path+filename, header=0, index_col=0)
            
            self.data['transport_technologies'][technologyName][dataType] =\
                file
            
def nodes(self):

    path = self.paths['nodes']['folder']
    
    data_type = 'nodes'
    fileformat = 'csv'            
    filename = '{}.{}'.format(data_type, fileformat)  
    
    file = pd.read_csv(path+filename, header=0, index_col=False)
    
    indexNodes = np.arange(0, file['node'].values.size)
    
    for indexNode in indexNodes:
        self.data[data_type][file['node'].iloc[indexNode]] = indexNode
        
def times(self):
    
    path = self.paths['times']['folder']
    
    data_type = 'times'
    fileformat = 'csv'            
    filename = '{}.{}'.format(data_type, fileformat)  
    
    file = pd.read_csv(path+filename, header=0, index_col=False)
    
    indexNodes = np.arange(0, file['time'].values.size)
    
    for indexNode in indexNodes:
        self.data[data_type][file['time'].iloc[indexNode]] = indexNode  
        
def scenarios(self):
    
    path = self.paths['scenarios']['folder']
    
    data_type = 'scenarios'
    fileformat = 'csv'            
    filename = '{}.{}'.format(data_type, fileformat)  
    
    file = pd.read_csv(path+filename, header=0, index_col=False)
    
    indexNodes = np.arange(0, file['scenario'].values.size)
    
    for indexNode in indexNodes:
        self.data[data_type][file['scenario'].iloc[indexNode]] = indexNode         
        
                
                             
        
    