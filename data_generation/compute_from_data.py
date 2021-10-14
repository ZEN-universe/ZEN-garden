# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import numpy as np
import pandas as pd
    
def DistanceMtx(N, idx_arr, x_arr, y_arr):
    """
    Compute a matrix containing the distance between any two points in the
    domain referring to their indices in the flattened array
    
    """
    
    def f_eucl_dist(P0, P1):
        """
            Compute the Eucledian distance of two points in 2D
        """
        return ((P0[0] - P1[0])**2 + (P0[1] - P1[1])**2)**0.5    

    dist = np.zeros([N, N])    
    
    for idx0 in idx_arr:
        for idx1 in idx_arr:
            P0 = (x_arr[idx0], y_arr[idx0])
            P1 = (x_arr[idx1], y_arr[idx1])
        
            dist[idx0, idx1] = f_eucl_dist(P0, P1)  
            
    return dist
        
path_in = "..//data//network//"
file_in = pd.read_csv(path_in+'file.csv', header=0, index_col=None)

N = file_in['nodes'].size
idx_arr = np.arange(N, dtype=np.int)
x_arr = file_in['X'].values
y_arr = file_in['Y'].values

dist = DistanceMtx(N, idx_arr, x_arr, y_arr)

path_out = path_in
file_out = pd.DataFrame(dist, columns=idx_arr, index=idx_arr)
file_out.to_csv(path_out+'eucleadian_distance.csv',\
    header=True, index=True)