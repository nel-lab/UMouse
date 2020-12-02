# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:38:04 2020
Script for plotting different decomposition and clustering analyses of Richard Warren's
locomotion data. Data involves a headfixed animal on a running wheel with an occasional
obstacle and reward. 

@author: Jake
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from behavelet import wavelet_transform
from numpy import genfromtxt
from sklearn.decomposition import PCA, NMF, KernelPCA, FastICA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from mpl_toolkits import mplot3d

# Set path
data_dir = 'D:/data/Behavior data/RW_data/'

#%% access the data
trackingDf = pd.read_csv('D:/data/Behavior data/RW_data/trackingDf.csv')
pawsRS = genfromtxt('D:/data/Behavior data/RW_data/trackingArray.csv', delimiter=',')

#%% wavelet analysis
#freqs, power, X_new = wavelet_transform(pawsRS[1:, :], n_freqs=25, fsample=250., fmin=1., fmax=50.)
#np.savetxt("D:/data/Behavior data/RW_data/freqsArray.csv", freqs, delimiter=",")
#np.savetxt("D:/data/Behavior data/RW_data/powerArray.csv", power, delimiter=",")
#np.savetxt("D:/data/Behavior data/RW_data/X_newArray.csv", X_new, delimiter=",")

#%% OPTIONAL - shortcut if you have already performed behavelet
freqs = pd.read_csv('D:/data/Behavior data/RW_data/freqsArray.csv')
power = pd.read_csv('D:/data/Behavior data/RW_data/powerArray.csv')
X_new = pd.read_csv('D:/data/Behavior data/RW_data/X_newArray.csv')

#%% compare decomposition methods

comp = 3

p = PCA(n_components=comp)
n = NMF(n_components=comp, max_iter=500, alpha=200)
t = TSNE(n_components=comp-1)
k = KernelPCA(n_components=comp, kernel='poly')
f = FastICA(n_components=comp, max_iter=1000)

downsample=50

pca  = p.fit_transform(X_new[::downsample])
nmf  = n.fit_transform(X_new[::downsample])
tsne = t.fit_transform(X_new[::downsample])
kpca = k.fit_transform(X_new[::downsample])
fica = f.fit_transform(X_new[::downsample])

dim_red = [pca,nmf,tsne,kpca,fica]

name = ["pca","nmf","tsne","kpca","fica"]

#save frame IDs for each downsampled value
PCA_points_ix = np.array(list(range(0, len(X_new), downsample)))

#%% Separate obstacle times into early and late

obstDiff = np.diff(trackingDf['obstacleBool'])

obstStart = np.where(obstDiff==1)[0] + 1 #add 1 to adjust index for np.diff
obstEnd = np.where(obstDiff==-1)[0] + 1

obstEarly = np.zeros([1,len(trackingDf['obstacleBool'])])
obstMid   = np.zeros([1,len(trackingDf['obstacleBool'])])
obstLate  = np.zeros([1,len(trackingDf['obstacleBool'])])

n_div = 3

#Make separate indeces for early, middle, and late obstacle times
for bout in range(0, len(obstStart)):
    assert obstStart[bout] < obstEnd[bout]
    
    obstDur = obstEnd[bout] - obstStart[bout]
    #make sure that the duration is divisible by the number of groups. 
    if obstDur % 3 != 0: #trim off the front end of the duration index until no remainder
        obstInd = np.array(range(obstStart[bout] + (obstDur % 3), obstEnd[bout])) 
    else:
        obstInd = np.array(range(obstStart[bout], obstEnd[bout]))
    
    splitInds = np.split(obstInd, n_div)
    
    #hardcoded this bit for 3 groups but could be moded for any n groups
    obstEarly[0][splitInds[0]] = 1
    obstMid[0][splitInds[1]] = 1
    obstLate[0][splitInds[2]] = 1 
    
#make labels for each principal component data point
PC_labels = np.zeros([1,len(pca)])
# 1 = reward
PC_labels[0, [np.where(trackingDf.rewardBool[PCA_points_ix] ==1)]] = 1

# 2= early obstacle   3 = mid obstacle   4 = late obstacle
PC_labels[0, [np.where(obstEarly.T[PCA_points_ix] ==1)]] = 2
PC_labels[0, [np.where(obstMid.T[PCA_points_ix] ==1)]] = 3
PC_labels[0, [np.where(obstLate.T[PCA_points_ix] ==1)]] = 4

#%% PLOT 3D color coded PCs
scores_plot = pca
fig = plt.figure()
ax = plt.axes(projection='3d', title='First 3 PCs for behavelet locomotion data')

#ax.scatter(*scores.T[:,[np.where(PC_labels==0)[1]]], c='r', marker='o', alpha=0.05) #other
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==1)[1]]], c='c', marker='o', alpha=0.2) #reward
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==2)[1]]], c='g', marker='o', alpha=0.2) #obst1/3
#ax.scatter(*scores_plot.T[:,[np.where(PC_labels==3)[1]]], c='b', marker='o', alpha=0.2)
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==4)[1]]], c='r', marker='o', alpha=0.2)

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')

pickle.dump(fig, open('FigureObject.fig.pickle', 'wb')) 
#modify kernels, alpha and gamma for kpca
#try tsne in 2 dimensions
#prioritize getting movement traces
#

#%% 2 dim plotting for tsne
scores_plot = tsne
fig = plt.figure()
ax = plt.axes(title='First dims for tsne for behavelet locomotion data')

#ax.scatter(*scores.T[:,[np.where(PC_labels==0)[1]]], c='r', marker='o', alpha=0.05) #other
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==1)[1]]], c='c', marker='o', alpha=0.2) #reward
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==2)[1]]], c='g', marker='o', alpha=0.2) #obst1/3
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==3)[1]]], c='b', marker='o', alpha=0.2)
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==4)[1]]], c='r', marker='o', alpha=0.2)

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')

# %% and try nmf and feature agglomeration and random projection 
# or dictionary learning.    The regularization in nmf may help reduce spurious correlations. Try manipulating alpha during nmf.
# *kPCA* probably with radial basis function
# t-sne
# fast ICA (before or after)

#checkout sklearn for clustering algos and look up matrix decomposition. 
#%% 
#train an svm as a classifier

















#%%
#import pylab as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(*scores[800::10].T, c='c', marker='o', alpha=0.1)
ax.scatter(*scores[50:400:10].T, c='r', marker='o')
ax.scatter(*scores[400:600:10].T, c='b', marker='o')
ax.scatter(*scores[600:800:10].T, c='g', marker='o')

#%%
fig = plt.figure()
#ax = fig.add_axes()
plt.scatter(*scores[800::10].T, c='c', marker='o', alpha=0.1)
plt.scatter(*scores[50:400:10].T, c='r', marker='o')
plt.scatter(*scores[150000:150400:10].T, c='b', marker='o')
plt.scatter(*scores[300000:300400:10].T, c='g', marker='o')


#%%
fig = plt.figure()
plt.plot(scores*100)
#%%
plt.figure()
plt.imshow(new.values[:,:].T, aspect='auto')








