"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to read the data from input files, collect them into a dictionary.
==========================================================================================================================================================================="""

import pandas as pd
import numpy as np
from deepdiff import DeepDiff
import sys
import logging

class Read:
    
    def __init__(self):
        pass

    def carriers(self):
        
        logging.info('read data of all the carriers')   
        
        self.carrierTypes = {'input_carriers':'setCarriersIn',
                             'output_carriers':'setCarriersOut',
                             }
        
        self.dataTypes = {'input_carriers': ['availability', 'importPrice', 'exportPrice'],
                     'output_carriers':['demand', 'importPrice', 'exportPrice']}
            
        for carrierType in self.carrierTypes.keys():
            
            for carrierName in self.data[carrierType].keys():
                
                # Read the input data of the energy carrier
                for data_type in self.dataTypes[carrierType]:
                    
                    path = self.paths[carrierType][carrierName]['folder']
                    
                    fileformat = 'csv'            
                    filename = '{}.{}'.format(data_type, fileformat)
                            
                    # table attributes                     
                    file = pd.read_csv(path+filename, header=0, index_col=None) 
                    indexList = ['scenario', 'time', 'node']
                
                    self.data[carrierType][carrierName][data_type] = file.set_index(indexList)
            
    def technologies(self):
        
        logging.info('read data of all the technologies')      
        
        for technologyName in self.data['production_technologies'].keys():
            
            # Read the input data of the energy carrier
            for data_type in ['attributes', 'availability_matrix']:
                
                path = self.paths['production_technologies'][technologyName]['folder']
                
                fileformat = 'csv'            
                filename = '{}.{}'.format(data_type, fileformat)
                        
                # table attributes                     
                file = pd.read_csv(path+filename, header=0, index_col=0)
                
                self.data['production_technologies'][technologyName][data_type] =\
                    file
        
        for technologyName in self.data['storage_technologies'].keys():
            
            # Read the input data of the energy carrier
            for data_type in ['attributes', 'availability_matrix',\
                              'max_capacity','min_capacity']:
                
                path = self.paths['storage_technologies'][technologyName]['folder']
                
                fileformat = 'csv'            
                filename = '{}.{}'.format(data_type, fileformat)
                        
                # table attributes                     
                file = pd.read_csv(path+filename, header=0, index_col=0)
                
                self.data['storage_technologies'][technologyName][data_type] =\
                    file   
    
        for technologyName in self.data['transport_technologies'].keys():
            
            # Read the input data of the energy carrier
            for data_type in ['availability_matrix',\
                              'cost_per_distance', 'efficiency_per_distance']:
                
                path = self.paths['transport_technologies'][technologyName]['folder']
                
                fileformat = 'csv'            
                filename = '{}.{}'.format(data_type, fileformat)
                        
                # table attributes                     
                file = pd.read_csv(path+filename, header=0, index_col=0)
                
                self.data['transport_technologies'][technologyName][data_type] =\
                    file   
                
    def nodes(self):
    
        path = self.paths['nodes']['folder']
        
        data_type = 'nodes'
        fileformat = 'csv'            
        filename = '{}.{}'.format(data_type, fileformat)  
        
        file = pd.read_csv(path+filename, header=0, index_col=False)
        
        indexNodes = np.arange(0, file['nodes'].values.size)
        
        for indexNode in indexNodes:
            self.data['nodes'][file['nodes'].iloc[indexNode]] = indexNode
            
    def times(self):
        
        path = self.paths['times']['folder']
        
        data_type = 'times'
        fileformat = 'csv'            
        filename = '{}.{}'.format(data_type, fileformat)  
        
        file = pd.read_csv(path+filename, header=0, index_col=False)
        
        indexNodes = np.arange(0, file['times'].values.size)
        
        for indexNode in indexNodes:
            self.data['times'][file['times'].iloc[indexNode]] = indexNode  
            
    def scenarios(self):
        
        path = self.paths['scenarios']['folder']
        
        data_type = 'scenarios'
        fileformat = 'csv'            
        filename = '{}.{}'.format(data_type, fileformat)  
        
        file = pd.read_csv(path+filename, header=0, index_col=False)
        
        indexNodes = np.arange(0, file['scenarios'].values.size)
        
        for indexNode in indexNodes:
            self.data['scenarios'][file['scenarios'].iloc[indexNode]] = indexNode      
        
                
                             
        
    