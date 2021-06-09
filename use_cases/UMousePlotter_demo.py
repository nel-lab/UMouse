#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:17:28 2021

@author: jimmytabet
"""

import pandas as pd
import numpy as np
import umouse.UMousePlotter_functions as UMPlot

#%% random test example
a = np.random.rand(10,3)
b = np.random.rand(10,3)
c = np.random.rand(10,3)
a_df = pd.DataFrame(a, columns = ['dim1','dim2','dim9'])
b_df = pd.DataFrame(b, columns = ['dim1','dim2','dim9'])
c_df = pd.DataFrame(c, columns = ['dim1','dim2','dim9'])

# UMAP_dfs = [a_df]
UMAP_dfs = [a_df, b_df, c_df]

UMAP_dfs_all = pd.concat(UMAP_dfs)
behavior_labels = [np.random.randint(0,4,len(i)) for i in UMAP_dfs]
# behavior_labels = np.random.randint(0,4,len(UMAP_dfs_all))
behavior_legend = [f'beh_{num}' for num in range(max(np.unique(behavior_labels))+1)]

n = 3
k = 3
spread=3

UMPlot.plot_embedding(UMAP_dfs, sep_data=True, save='random/UMAP_embedding')
inter = UMPlot.interactive(n,k,spread,UMAP_dfs, behavior_labels, behavior_legend)
inter.get_points(save_embedding='random/UMAP_beh', save_chosen_points='random/points')
inter.plot_traces(UMAP_dfs, ['dim9'], save='random/traces')

#%% example with 3 camera set up
UMAP_dfs = pd.read_csv('3_camera/nmf.csv')
behavior_dfs = '3_camera/behavior.csv'
behavior_variable = ['Lpaw_Y', 'Rpaw_Y']
n = 3
k = 5
spread=70
fps = 70
ds = 50
local_path_to_video = '/Users/jimmytabet/NEL/Projects/BH3D/mov.h5'
video_path = local_path_to_video
save_path = '3_camera/behavior_montage.avi'

inter = UMPlot.interactive(n, k, spread, UMAP_dfs, ds=50)
inter.get_points(save_embedding=True, save_chosen_points=True)
inter.plot_traces(behavior_dfs, behavior_variable, save=True)
inter.behavior_montage(video_path, save_path, fps)