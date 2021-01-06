# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:19:27 2021

@author: Jake
"""
#%% Load modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

#%% take spectrographic data and downsample by 10

X_new = pd.read_csv('D:/data/Behavior data/RW_data/X_newArray.csv')
downsamp = 10
XDSamp = X_new[::downsamp]

#%% Perform k-means clustering on spectrographic sample

clus=15  #arbitrarily picking 15 clusters
Cluster = KMeans(n_clusters = clus)
kmeans = Cluster.fit(XDSamp)
y_kmeans = kmeans.labels_ #tsneKmeans.predict(tsne)

#identfy the number of data points in each cluster
fig = plt.figure() 
plt.hist(y_kmeans)
plt.title('num clusters = ' + str(clus))

#%%   incomplete code I made to plot the mean spectrogram of each cluster after the kmeans

# plot average spectrogram from each cluster
#
#fig, axes = plt.subplots(ncols=4, nrows=math.ceil(clus/4), figsize=(9, 3), sharex=True, sharey=True)
#ax = axes.ravel()
#
#for thisAx in np.unique(y_kmeans):
#    thisClus = XDSamp[y_kmeans == thisAx]
#    ax[thisAx].imshow()
#
#ax[0].imshow(fgray, cmap=plt.cm.gray)
#ax[0].set_title('Input image')
#ax[1].imshow(-dist_transform, cmap=plt.cm.gray)
#ax[1].set_title('Distance transform')
#ax[2].imshow(segmented, cmap=plt.cm.nipy_spectral)
#ax[2].set_title('Separated objects')
#
#for a in ax:
#    a.set_axis_off()
#
#fig.tight_layout()
#plt.show()

#%% Sample uniformly from each cluster

#find min number of samples in smallest cluster
minbin = plt.hist(y_kmeans)

#samplme that number from each bin

np.random.choice(aa, size=minbin, replace=False)

#%% Use samples from clusters as input to t-sne to generate a 2D embedding

#%% Map remaining data onto the embedding space

#%% Perform watershed transformation to cluster the data? 

#%% Plot traces and videos from each cluster in order to label them

#%% Generate a labelled behavior map in 2D 