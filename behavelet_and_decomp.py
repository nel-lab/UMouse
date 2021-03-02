# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:38:04 2020

To be run after open_and_format_RW_data.py

Script for plotting different decomposition and clustering analyses of Richard Warren's
locomotion data. Data involves a headfixed animal on a running wheel with an occasional
obstacle and reward. 

modified Jimmy's code for decomposition comparison, n nearest neighbors, and plotting trajectories 
so that it works with Richard's data. 

@author: Jake
"""

#select the dataset to analyze
data_fn = 'trackingData_201115_000'
#data_fn = 'trackingData_181215_003'
expt_fn = data_fn[-10:]

#set path
data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'

#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.decomposition import PCA, NMF, KernelPCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
import cv2

#%% access the data

trackingDf = pd.read_csv(data_dir + expt_fn + '_Df.csv') 
pawsRS = genfromtxt(data_dir + expt_fn + '_pawsArray.csv', delimiter=',') 

#%% Morlet wavelet analysis

#load module and perform transformation
from behavelet import wavelet_transform
freqs, power, X_new = wavelet_transform(pawsRS, 
                                        n_freqs=25, 
                                        fsample=250., 
                                        fmin=1., 
                                        fmax=50.)

#save variables for use later
np.savetxt(data_dir + expt_fn + '_mwtFreqs.csv', freqs, delimiter=",")
np.savetxt(data_dir + expt_fn + '_mwtPower.csv', power, delimiter=",")
np.savetxt(data_dir + expt_fn + '_mwtXNew.csv', X_new, delimiter=",")

#%% MWT but with additional features jawVars, whiskerAngle, and bodyAngles

from behavelet import wavelet_transform

#format the data
pawBodyJaw = np.concatenate((pawsRS, 
                             trackingDf['bodyAngles'].to_numpy().reshape(403912,1), 
                             trackingDf['jawVarX'].to_numpy().reshape(403912,1)), 
                            axis=1)

#perform MWT
freqs, power, X_new = wavelet_transform(pawBodyJaw, 
                                        n_freqs=25, 
                                        fsample=250., 
                                        fmin=1., 
                                        fmax=50.)

#save variables for use later
np.savetxt(data_dir + expt_fn + '_jawBod_mwtFreqs.csv', freqs, delimiter=",")
np.savetxt(data_dir + expt_fn + '_jawBod_mwtPower.csv', power, delimiter=",")
np.savetxt(data_dir + expt_fn + '_jawBod_mwtXNew.csv', X_new, delimiter=",")

#%%  plot the behavelet data
 
#plot spectropgrahic data with vertical lines indicating whisker contact and reward
wiskArray = np.array(np.where(trackingDf['wiskContTimeBool'])[0])
rewardArray = np.array(np.where(trackingDf['rewardBool'])[0])

fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax1.imshow(X_new[:50000,0:350].T, aspect='auto')
#Whisker contact times (only plotting the first frame of contact sequence)
for wiskInd,thisWisk in enumerate(wiskArray):
    if wiskInd==0:
        ax1.axvline(x=thisWisk, color = 'w', lw=0.5)
    elif thisWisk > 50000:
        break
    elif thisWisk - wiskArray[wiskInd-1]>1:
        ax1.axvline(x=thisWisk, color = 'w', lw=0.5)
#Reward
for rewInd,thisRew in enumerate(rewardArray):
    if rewInd==0:
        ax1.axvline(x=thisRew, color = 'b', lw=0.5)
    elif thisRew > 50000:
        break
    elif thisRew - rewardArray[rewInd-1]>1:
        ax1.axvline(x=thisRew, color = 'b', lw=0.5)
     
#for plotting without the indactor lines
# plt.imshow(X_new[:50000,:].T, aspect='auto')
# #plt.imshow(X_new.values[:50000,:].T, aspect='auto')
# plt.axvline(x=np.where(trackingDf['wiskContTimeBool'][:50000]), color='w')

plt.title('Behavelet output ' + expt_fn + ' white=whisker, blue=reward')
plt.ylabel('Paws * dimensions')
plt.xlabel('frame # at 250Hz')
plt.savefig(data_dir + expt_fn + 'spectImg')

#%% OPTIONAL - shortcut if you have already performed behavelet

#without jaw angle and body angle
#X_new = pd.read_csv(data_dir + expt_fn + '_mwtXNew.csv')
#with jaw angle and body angle
#X_new = pd.read_csv(data_dir + expt_fn + '_jawBod_mwtXNew.csv')

#X_new = X_new.to_numpy() #from pandas series

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
    
# #make labels for each principal component data point
# PC_labels = np.zeros([1,len(pca)])
# # 1 = reward
# PC_labels[0, [np.where(trackingDf.rewardBool[PCA_points_ix] ==1)]] = 1

# # 2= early obstacle   3 = mid obstacle   4 = late obstacle
# PC_labels[0, [np.where(obstEarly.T[PCA_points_ix] ==1)]] = 2
# PC_labels[0, [np.where(obstMid.T[PCA_points_ix] ==1)]] = 3
# PC_labels[0, [np.where(obstLate.T[PCA_points_ix] ==1)]] = 4

#Make behavior label for un-downsampled data
# 1=reward  2= early obstacle   3 = mid obstacle   4 = late obstacle
PC_labels = np.zeros([1,len(pawsRS)])
PC_labels[0, [np.where(trackingDf.rewardBool ==1)]] = 1
PC_labels[0, [np.where(obstEarly.T ==1)]] = 2
PC_labels[0, [np.where(obstMid.T ==1)]] = 3
PC_labels[0, [np.where(obstLate.T ==1)]] = 4

np.savetxt(data_dir + expt_fn + "_bxLabelsArray.csv", 
           PC_labels, 
           delimiter=",")

#%% Mostly older code from here on out

#%% 


#%% PLOT 3D color coded PCs

scores_plot = pca

fig = plt.figure()
ax = plt.axes(projection='3d', title='pca First 3 PCs for behavelet locomotion data.')

#ax.scatter(*scores_plot.T[:,[np.where(PC_labels==0)[1]]], c='r', marker='o', alpha=0.05) #other
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==1)[1]]], c='c', marker='o', alpha=0.2) #reward
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==2)[1]]], c='g', marker='o', alpha=0.2) #obst1/3
#ax.scatter(*scores_plot.T[:,[np.where(PC_labels==3)[1]]], c='b', marker='o', alpha=0.2)
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==4)[1]]], c='k', marker='o', alpha=0.2)

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')

#%% TSNE-2dim, Gaussian filter, and kmeans clustering

#treat the heatmap like a 3d image, sort of like a mountain plot. 
tsneGaussCoords = []
for Xpix in range(tsneImgGauss.shape[0]):
    for Ypix in range(tsneImgGauss.shape[1]):
        if tsneImgGauss[Xpix,Ypix] > 0:
            tsneGaussCoords.append([Xpix, Ypix, tsneImgGauss[Xpix,Ypix]])
tsneGaussCoords = np.array(tsneGaussCoords)
tsneGaussCoordsNorm = tsneGaussCoords
tsneGaussCoordsNorm[:,2] *= 100

#run kmeans clustering and get labels 
clus=9
tsneCluster = KMeans(n_clusters = clus)
tsneKmeans = tsneCluster.fit(tsneGaussCoords)
y_kmeans = tsneKmeans.labels_ #tsneKmeans.predict(tsneGaussCoords)

#plot as a scatterplot 
fig_all = plt.figure()
ax = plt.axes(title='tsne, gaussian, kmeans clustering. clusters='+str(clus))
plt.scatter(tsneGaussCoords[:,0], tsneGaussCoords[:,1], c=y_kmeans, s=50, alpha=0.4, cmap='viridis')
ax.scatter(*tsneKmeans.cluster_centers_.T, c='k', marker='x')


