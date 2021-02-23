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

trackingDf = pd.read_csv(data_dir + expt_fn + '_Df.csv') #'D:/data/BehaviorData/RW_data/trackingDf.csv'
pawsRS = genfromtxt(data_dir + expt_fn + '_pawsArray.csv', delimiter=',') #'D:/data/BehaviorData/RW_data/trackingArray.csv', delimiter=','

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
 
plt.figure()
plt.imshow(X_new[:50000,:].T, aspect='auto')
#plt.imshow(X_new.values[:50000,:].T, aspect='auto')
plt.title('Behavelet output ' + expt_fn)
plt.ylabel('Paws * dimensions')
plt.xlabel('frame # at 250Hz')

#%% OPTIONAL - shortcut if you have already performed behavelet
#freqs = pd.read_csv('D:/data/BehaviorData/RW_data/freqsArray.csv')
#power = pd.read_csv('D:/data/BehaviorData/RW_data/powerArray.csv')
#X_new = pd.read_csv('D:/data/BehaviorData/RW_data/X_newArray.csv')

#%% compare decomposition methods

# comp = 2
# p = PCA(n_components=comp)
# n = NMF(n_components=comp, max_iter=500, alpha=2)
# t = TSNE(n_components=comp-1)
# k = KernelPCA(n_components=comp)
# f = FastICA(n_components=comp, max_iter=1000)

# downsample=50

# pca  = p.fit_transform(X_new[::downsample])
# nmf  = n.fit_transform(X_new[::downsample])
# tsne = t.fit_transform(X_new[::downsample])
# kpca = k.fit_transform(X_new[::downsample])
# fica = f.fit_transform(X_new[::downsample])

# #save frame IDs for each downsampled value
# PCA_points_ix = np.array(list(range(0, len(X_new), downsample)))

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

#%% Mostly older code from here on out

#%% 


#%% Run kmeans on the pca output

clus = 6

cluster = KMeans(n_clusters = clus)
kmeans = cluster.fit(pca)  #labels_    cluster_centers_
    
fig_all = plt.figure()
ax = plt.axes(projection='3d', title='kmeans clustering of pca data. clusters='+str(clus))

y_kmeans = kmeans.predict(pca)
plt.scatter(x=pca[:,0], y=pca[:,1], zs=pca[:,2], c=y_kmeans, s=50, alpha=0.05, cmap='viridis')
ax.scatter(*kmeans.cluster_centers_.T, c='k', marker='x')

#%% try a gaussian low pass filter on tsne output
tsneRound = tsne.astype(int)
tsneMin = np.min(tsneRound, axis=0)
tsneMax = np.max(tsneRound, axis=0)

tsneNonNeg = tsneRound
tsneNonNeg[:,0] += abs(tsneMin[0])
tsneNonNeg[:,1] += abs(tsneMin[1])

tsneImg = np.zeros([160,160]) 
for pixInd in range(0,len(tsneNonNeg)): 
    tsneImg[tsneNonNeg[pixInd,0], tsneNonNeg[pixInd,1]] += 1

tsneImgGauss = cv2.GaussianBlur(tsneImg, (51,51), 0)

fig = plt.figure()
plt.imshow( tsneImgGauss )
plt.title('tsne outputs passed through gaussian filter (51,51)')
plt.colorbar()

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

#%% TSNE 2dim and kmeans without filter

#run kmeans
clus=4
tsneCluster = KMeans(n_clusters = clus)
tsneKmeans = tsneCluster.fit(tsne)
y_kmeans = tsneKmeans.labels_ #tsneKmeans.predict(tsne)

#plot outputs colorcoded by cluster ID
fig_all = plt.figure()
ax = plt.axes(title='tsne/kmeans clustering. clusters='+str(clus))
plt.scatter(tsne[:,0], tsne[:,1], c=y_kmeans, s=50, alpha=0.2, cmap='viridis')
ax.scatter(*tsneKmeans.cluster_centers_.T, c='k', marker='x')

#%% kclosest points to centroids function

def kclosest(k, dim, cluster_centroids):
    dist = []

    for row in dim:
        axis_diff = row-cluster_centroids 
        dist.append(np.linalg.norm(axis_diff, axis=1))
   
    dist = np.array(dist)
    closest = np.argsort(dist, axis=0)
    ids = closest[:k]
        
    return dist, closest, ids

#%% kclosest points
k = 5

kmeans_dist, kmeans_closest, kmeans_ids = kclosest(k, pca, kmeans.cluster_centers_)

#%% plot traces around kclosest points - paw center 'Z'
spread=200   # plot k closest frames (+- spread)
paw_axis = 'Y'
paws = [col for col in trackingDf.columns if col[-1] == paw_axis]

num_clus = kmeans_ids.shape[1] 

fig, ax = plt.subplots(nrows=num_clus, ncols=kmeans_ids.shape[0])   
fig.suptitle('pca,'+ ' kmeans, ' + 'paw_axis='+paw_axis, size='x-large') 

for clus in range(num_clus):           
     for num, i in enumerate(kmeans_ids[:,clus]): 
                              
         ax[clus,num].plot(trackingDf.loc[downsample*i-spread:downsample*i+spread+1,paws])  
         
         if num==0:                   
             ax[clus,num].set_ylabel('cluster '+str(clus+1), size='large')  
         if clus==0:
             ax[clus,num].set_title('closest: '+str(num+1))
          
DPI = fig.get_dpi()
fig.set_size_inches(1920.0/float(DPI),1080.0/float(DPI))

fig.tight_layout(rect=[0, 0.05, 1, 0.95]) 

# fig.savefig('/Users/jimmytabet/Desktop/Behavioral Classification Results/Std. Before/traces/'+dic['name']+'/'+c_type, dpi=DPI, bbox_inches='tight')
#plt.close('all')








