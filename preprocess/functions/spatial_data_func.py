"""=====================================================================================================================
Title:        ZEN-GARDEN
Created:      April-2022
Authors:      Johannes Burger (jburger@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Simple functions to help with creating, extracting or processing spatial data into the required format.
======================================================================================================================"""

import pandas as pd
import numpy as np


def calc_distance(start, end, node_df):
    d_x = node_df[node_df['node'] == start]['x'].values[0] - node_df[node_df['node'] == end]['x'].values[0]
    d_y = node_df[node_df['node'] == start]['y'].values[0] - node_df[node_df['node'] == end]['y'].values[0]
    return np.sqrt(d_x ** 2 + d_y ** 2)

def get_connection_df(connection_list, node_df):
    ls = []


