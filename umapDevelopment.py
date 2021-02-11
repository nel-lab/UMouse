# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:07:25 2021

@author: Jake
"""


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datashader as ds
import colorcet as cc
import umap
import umap.plot
from scipy.stats import gaussian_kde
from mpl_toolkits import mplot3d

#%% Load spctrographic data and downsample

X_new = pd.read_csv('D:/data/Behavior data/RW_data/X_newArray.csv')
downsamp = 25
XDSamp = X_new[::downsamp]

#save frame IDs for each downsampled value
points_ix = np.array(list(range(0, len(X_new), downsamp)))

#%% save/load DS 25 dataset

#np.savetxt("D:/data/Behavior data/RW_data/behaveletOutDS" + str(downsamp) + '.csv', XDSamp, delimiter=",")

#downsamp = 25
#XDSamp = pd.read_csv('D:/data/Behavior data/RW_data/behaveletOutDS' + str(downsamp) + '.csv')

#%% Try adding positional data to the spectrographic data for umap input

#from numpy import genfromtxt
pawsRS = genfromtxt('D:/data/Behavior data/RW_data/trackingArray.csv', delimiter=',')
pawsRS = np.delete(pawsRS, [0,-1], 0)
pawsRS = pawsRS[::downsamp]
#np.concatenate((XDSamp, pawsRS),1)

#%% Use pawsRS to add location features to umap embedding inputs

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)

#mapper = reducer.fit(XDSamp)
mapper = reducer.fit(np.concatenate((XDSamp, pawsRS),1))

embedding = mapper.embedding_


#%% try some plotting

fig = plt.figure()
umap.plot.points(mapper)
plt.title('spectrographic and position data input. DS=25')

#%% 3D plotting 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#color will be equal to position along the z axis for easier viewing
ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=embedding[:,2]/9, s=0.1)
#ax.view_init(-90, 35) #alter to change viewing angle (elevation, azimuth)

#%%   Gaussian filter for umap output

#kde_and_plot.py

#%% load behavior data and behavior labels 

trackingDf = pd.read_csv('D:/data/Behavior data/RW_data/trackingDf.csv')
pd.read_csv('D:/data/Behavior data/RW_data/bx_label_array.csv')
bx_labels = PC_labels

#%% 2 dim plotting for umap. converted from tsne 2dim plotting 
# # 0=other, 1=reward+500ms, 2=obst1/3,   3=obstMid,   4=obstEnd
# scores_plot = embedding
# fig = plt.figure()
# ax = plt.axes(title='UMAP cyan=reward, obstacle=gbk, red=other')

# ax.scatter(*scores_plot.T[:,[np.where(bx_labels==0)[1]]], c='r', marker='o', alpha=0.05) #other
# ax.scatter(*scores_plot.T[:,[np.where(bx_labels==1)[1]]], c='c', marker='o', alpha=0.2) #reward
# ax.scatter(*scores_plot.T[:,[np.where(bx_labels==2)[1]]], c='g', marker='o', alpha=0.2) #obst1/3
# ax.scatter(*scores_plot.T[:,[np.where(bx_labels==3)[1]]], c='b', marker='o', alpha=0.2)
# ax.scatter(*scores_plot.T[:,[np.where(bx_labels==4)[1]]], c='k', marker='o', alpha=0.2)

# ax.set_xlabel('Dim 1')
# ax.set_ylabel('Dim 2')





