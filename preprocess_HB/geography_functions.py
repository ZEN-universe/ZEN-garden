"""=====================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      September-2022
Authors:      Johannes Burger (jburger@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Small helper functions for anything related to geopgraphical nodes, edges, etc.
====================================================================================================================="""


import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import geopandas as gpd


def edges_from_nodes(dataset=None):
    """
    Creates a .csv file with all possible edges from the nodes specified in the given dataset
    :param dataset: Name of the dataset to be used and which is located in the data folder
    :return df: Dataframe with all possible edges
    """
    if not dataset:
        print("No dataset defined!")
        return None
    os.chdir('/Users/jburger/Documents/GitHub/ZEN-garden/')
    dataset = 'Pioneering_CCTS'
    nodes = list(pd.read_csv(f"./data/{dataset}/systemSpecification/setNodes.csv")['node'])
    ls_combinations = list(itertools.combinations(nodes, 2))
    ls_combinations.extend(list(itertools.combinations(nodes[::-1], 2)))
    df = pd.DataFrame(ls_combinations, columns=['nodeFrom', 'nodeTo'])
    df['edge'] = df['nodeFrom'] + '-' + df['nodeTo']
    df.sort_values('edge', inplace=True, ignore_index=True)
    df = df[['edge', 'nodeFrom', 'nodeTo']]

    return df


def plot_nodes_on_map(dataset=None):
    """
    Plots all nodes in the dataset on a map of Europe
    :param dataset: Name of dataset
    """
    os.chdir('/Users/jburger/Documents/GitHub/ZEN-garden/')
    nodes = pd.read_csv(f"./data/{dataset}/systemSpecification/setNodes.csv")
    nodes['x'] = nodes['x'].apply(lambda x: x * 1000)
    nodes['y'] = nodes['y'].apply(lambda x: x * 1000)

    # geometry = [Point(xy) for xy in zip(nodes["x"], nodes["y"])]
    geodata = gpd.GeoDataFrame(nodes, crs='epsg:32632', geometry=gpd.points_from_xy(nodes['x'], nodes['y']))
    geodata = geodata.to_crs(crs='epsg:3035')

    europe_shapefile = gpd.read_file('./preprocess_HB/NUTS_RG_01M_2021_3035_LEVL_0.shp')

    fig, ax = plt.subplots(figsize=(7, 7))
    europe_shapefile.plot(ax=ax, facecolor='Grey', edgecolor='k', alpha=1, linewidth=1, cmap="summer")
    # You can use different 'cmaps' such as jet, plasma ,magma, infereno,cividis, binary...(I simply chose cividis)
    geodata.plot(ax=ax, color='red', markersize=15)

    fig.suptitle('Nodes over Europe', fontsize=12)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize='medium')
    ax.set_xlim([3.5e6, 5.5e6])
    ax.set_ylim([2.3e6, 5e6])
    plt.show()


def test_node_in_edges(dataset=None):
    """Tests if each node has at least one edge connected to it and if all edges have nodes"""
    os.chdir('/Users/jburger/Documents/GitHub/ZEN-garden/')
    nodes = pd.read_csv(f"./data/{dataset}/systemSpecification/setNodes.csv")
    nodes = pd.Series(nodes['node'].unique())
    edges = pd.read_csv(f"./data/{dataset}/systemSpecification/setEdges.csv")
    edge_nodes = pd.Series(pd.concat([edges['nodeFrom'], edges['nodeTo']], ignore_index=True).unique())
    nodes.sort_values(inplace=True, ignore_index=True)
    edge_nodes.sort_values(inplace=True, ignore_index=True)
    if nodes.equals(edge_nodes):
        print('All edges and nodes are fine.')
        return
    else:
        print('No success! The following nodes do not have edges or edges exist that do not have nodes:')
        print(pd.concat([edge_nodes, nodes]).drop_duplicates(keep=False))


if __name__ == '__main__':
    data = 'Pioneering_CCTS'
    # plot_nodes_on_map('Pioneering_CCTS')
    # df_edges = edges_from_nodes('Pioneering_CCTS')
    test_node_in_edges(dataset=data)

