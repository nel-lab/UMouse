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

#%% Load spctrographic data and downsample

X_new = pd.read_csv('D:/data/Behavior data/RW_data/X_newArray.csv')
downsamp = 25
XDSamp = X_new[::downsamp]

#save frame IDs for each downsampled value
points_ix = np.array(list(range(0, len(X_new), downsamp)))

#%% 
reducer = umap.UMAP()

embedding = reducer.fit_transform(XDSamp)

mapper = reducer.fit(XDSamp)
#%% try some plotting

fig = plt.figure()
umap.plot.points(mapper)

#%%   Gaussian filter for umap output

embedding_df = pd.DataFrame(data = embedding)

#calculate buffer for borders
deltaX = (max(embedding_df.iloc[:,0]) - min(embedding_df.iloc[:,0]))/10
deltaY = (max(embedding_df.iloc[:,1]) - min(embedding_df.iloc[:,1]))/10

#calculate plotting min,max + buffer
xmin = embedding_df.iloc[:,0].min() - deltaX
xmax = embedding_df.iloc[:,0].max() + deltaX
ymin = embedding_df.iloc[:,1].min() - deltaY
ymax = embedding_df.iloc[:,1].max() + deltaY

#make a mesh grid on which to plot stuff
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

#make useful variables then calculate kde
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([embedding_df.iloc[:,0], embedding_df.iloc[:,1]])

# calculate kde from tsne 2dim data

bw_val = 0.2
kernel = gaussian_kde(values, bw_method = bw_val)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure()
plt.imshow(f)

# visualize kde plot

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('UMAP: 2D Gaussian KDE. BW=' + str(bw_val) + '. DS='+str(downsamp))

#%% load behavior data and behavior labels 

#trackingDf = pd.read_csv('D:/data/Behavior data/RW_data/trackingDf.csv')
#pd.read_csv('D:/data/Behavior data/RW_data/bx_label_array.csv')
#bx_labels = PC_labels

#%% 2 dim plotting for umap. converted from tsne 2dim plotting 
# 0=other, 1=reward+500ms, 2=obst1/3,   3=obstMid,   4=obstEnd
scores_plot = embedding
fig = plt.figure()
ax = plt.axes(title='UMAP cyan=reward, obstacle=gbk, red=other')

ax.scatter(*scores_plot.T[:,[np.where(bx_labels==0)[1]]], c='r', marker='o', alpha=0.05) #other
ax.scatter(*scores_plot.T[:,[np.where(bx_labels==1)[1]]], c='c', marker='o', alpha=0.2) #reward
ax.scatter(*scores_plot.T[:,[np.where(bx_labels==2)[1]]], c='g', marker='o', alpha=0.2) #obst1/3
ax.scatter(*scores_plot.T[:,[np.where(bx_labels==3)[1]]], c='b', marker='o', alpha=0.2)
ax.scatter(*scores_plot.T[:,[np.where(bx_labels==4)[1]]], c='k', marker='o', alpha=0.2)

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')





