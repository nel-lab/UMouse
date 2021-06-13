#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:17:28 2021

@author: jimmytabet
"""

#%% imports
import pandas as pd
import numpy as np
import umouse.UMousePlotter_functions as UMPlot

#%% random test example - set up data
# generate random embeddings
a_df = pd.DataFrame(np.random.rand(25,3), columns = ['dim1','dim2','dim3'])
b_df = pd.DataFrame(np.random.rand(25,3), columns = ['dim1','dim2','dim3'])
c_df = pd.DataFrame(np.random.rand(25,3), columns = ['dim1','dim2','dim3'])

# # single data set
# UMAP_dfs = [a_df]
# multiple data sets
UMAP_dfs = [a_df, b_df, c_df]

# generate random behavior labels
behavior_labels = [np.random.randint(0,4,len(i)) for i in UMAP_dfs]
behavior_legend = ['walk','run','jump','stop']

#%% random test example - manual points
# n: number of clusters, k: number of neighbors in cluster
n = 3
k = 3
spread = 3

# plot embedding
UMPlot.plot_embedding(UMAP_dfs, sep_data=True, save='random/UMAP_data_manual')

# interactive
inter = UMPlot.interactive(n, k, spread, UMAP_dfs)
# get points from random centroids
manual_pt = np.random.rand(n,3)
inter.get_points(manual_pt, sep_data=True)
# plot traces (using UMAP data as behavior data) 
inter.plot_traces(UMAP_dfs, ['dim3'], save='random/traces_manual')

#%% random test example - interactive
# remove 3rd dim to show interatice case
UMAP_dfs = [df.drop(columns = 'dim3') for df in UMAP_dfs]

# n: number of clusters, k: number of neighbors in cluster
n = 3
k = 3
spread = 3

# interactive
inter = UMPlot.interactive(n, k, spread, UMAP_dfs, behavior_labels, behavior_legend)
# get points
inter.get_points(save_embedding='random/UMAP_behavior_inter', save_chosen_points='random/points_inter')
# plot traces (using UMAP data as behavior data) 
inter.plot_traces(UMAP_dfs, ['dim2'], save='random/traces_inter')

#%% example with 3 camera set up
# set up data
UMAP_dfs = pd.read_csv('3_camera/nmf.csv')  # load from memory...
behavior_dfs = '3_camera/behavior.csv'      # or load from path
behavior_variable = ['Lpaw_Y', 'Rpaw_Y']    
n = 3
k = 5
spread=70
fps = 70
ds = 50
video_path = '/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/Behavior3D/mov.h5'
save_path = '3_camera/behavior_montage.avi'

# interactive
inter = UMPlot.interactive(n, k, spread, UMAP_dfs, ds=50)
# get points
inter.get_points(save_embedding=True, save_chosen_points=True)
# plot traces
inter.plot_traces(behavior_dfs, behavior_variable, save=True)
# make montage
inter.behavior_montage(video_path, save_path, fps)