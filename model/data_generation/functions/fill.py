"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Auxiliary methods used for data creation
==========================================================================================================================================================================="""

import os
import numpy as np
import pandas as pd

class Fill:
    
    def __init__(self, dictionary):
        
        self.pathMainFolder = '..//..//data//'
        self.dictionary = dictionary


    def distanceMatrix(self):
        
        xArr = self.dictionary['setNodes']['XCoord']
        yArr = self.dictionary['setNodes']['YCoord']
        N = len(self.dictionary['setNodes']['Names'])
        
        distance = self.computeDistanceMatrix(N, xArr, yArr, 'eucledian')
        
        for nameTransport in  self.dictionary['setTransport']:
            if 'pipeline' in nameTransport.split('_'):
                path = '{}//{}//{}//{}//'.format(self.pathMainFolder, self.dictionary['mainFolder'], 'setTransport', nameTransport)
                fileName = 'distance'
                ext = '.csv'
                df = pd.read_csv(path+fileName+ext, header=0, index_col=0)        
                df.loc[:,:] = distance
                df.to_csv(path+fileName+ext, header=True, index=True)
            
    def computeDistanceMatrix(self, N, xArr, yArr, distanceType):
        """
        Compute a matrix containing the distance between any two points in the
        domain referring to their indices in the flattened array
        
        """
        
        def f_eucl_dist(P0, P1):
            """
                Compute the Eucledian distance of two points in 2D
            """
            return ((P0[0] - P1[0])**2 + (P0[1] - P1[1])**2)**0.5    
        
        idx_arr = np.arange(N)
        dist = np.zeros([N, N])    
        
        for idx0 in idx_arr:
            for idx1 in idx_arr:
                P0 = (xArr[idx0], yArr[idx0])
                P1 = (xArr[idx1], yArr[idx1])
                
                if distanceType == 'eucledian':
                    dist[idx0, idx1] = f_eucl_dist(P0, P1) 
                else:
                    raise "Distance type not implemented"
                
        return dist