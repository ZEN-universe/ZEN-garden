# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import numpy as np


def DistanceMtx(self):
    """
    Compute a matrix containing the distance between any two points in the
    domain referring to their indices in the flattened array
    
    """
    
    def f_eucl_dist(P0, P1):
        """
            Compute the Eucledian distance of two points in 2D
        """
        return ((P0[0] - P1[0])**2 + (P0[1] - P1[1])**2)**0.5    
    
    N = self.input['Network']['size']
    dist = np.zeros([N, N])    
    
    x_arr = self.input['Network']['X']
    y_arr = self.input['Network']['Y']
    
    for idx0 in self.input['Network']['idx']:
        for idx1 in self.input['Network']['idx']:
            P0 = (x_arr[idx0], y_arr[idx0])
            P1 = (x_arr[idx1], y_arr[idx1])
        
            dist[idx0, idx1] = f_eucl_dist(P0, P1)  
            
    self.input['Network']['distance_mtx'] = dist
   
    
    
    
    