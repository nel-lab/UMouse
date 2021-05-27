# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:16:51 2020

t-sne development script 

@author: Jake
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import cv2
from scipy.stats import gaussian_kde

# Set path
data_dir = 'D:/data/Behavior data/RW_data/'

#%% OPTIONAL - shortcut if you have already performed behavelet

X_new = pd.read_csv('D:/data/Behavior data/RW_data/X_newArray.csv')
#trackingDf = pd.read_csv('D:/data/Behavior data/RW_data/trackingDf.csv')
#pawsRS = genfromtxt('D:/data/Behavior data/RW_data/trackingArray.csv', delimiter=',')

#%% Run tsne on all data points 

comp = 2  #n components/dimensions
downsamp = 25

#t = TSNE(n_components=comp, verbose = 2)
#tsne = t.fit_transform(X_new[::downsamp])

#np.savetxt("D:/data/Behavior data/RW_data/tsneOutDS" + str(downsamp) + ".csv", tsne, delimiter=",")

#%% get datapoint indeces post downsampling

#downsampInd = np.array(list(range(0, len(X_new), downsamp)))

#%% Visualize tsne

# fig = plt.figure()
# ax = plt.axes(title='2dim tsne downsample=' + str(downsamp))
# #ax.scatter(*tsne.T, marker='o', alpha=0.05) #other
# ax.scatter(x= tsne.iloc[:,0], y= tsne.iloc[:,1], marker='o', alpha=0.05) #other

#%% load 
tsne = pd.read_csv('D:/data/Behavior data/RW_data/tsneOutDS' + str(downsamp) + '.csv')

#%% preprocessing before  gaussian kde. 
#https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67

#calculate buffer for borders
deltaX = (max(tsne.iloc[:,0]) - min(tsne.iloc[:,0]))/10
deltaY = (max(tsne.iloc[:,1]) - min(tsne.iloc[:,1]))/10

#calculate plotting min,max + buffer
xmin = tsne.iloc[:,0].min() - deltaX
xmax = tsne.iloc[:,0].max() + deltaX
ymin = tsne.iloc[:,1].min() - deltaY
ymax = tsne.iloc[:,1].max() + deltaY

#make a mesh grid on which to plot stuff
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

#make useful variables then calculate kde
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([tsne.iloc[:,0], tsne.iloc[:,1]])

#%% calculate kde from tsne 2dim data

bw_val = 0.2
kernel = gaussian_kde(values, bw_method = bw_val)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure()
plt.imshow(f)

#%% visualize kde plot

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
plt.title('2D Gaussian Kernel density estimation. BW=' + str(bw_val) + '. DS='+str(downsamp))

#%% try different bandwidth values in the gaussian kde 

#bw_vals = [0.1, 0.15, 0.2, 0.25]
#flist = []
#
#for this_bw in bw_vals:
#    kernel = gaussian_kde(values, bw_method = this_bw)
#    f = np.reshape(kernel(positions).T, xx.shape)
#    #flist.append(f)
#    
#    fig = plt.figure(figsize=(8,8))
#    ax = fig.gca()
#    ax.set_xlim(xmin, xmax)
#    ax.set_ylim(ymin, ymax)
#
#    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
#    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
#    cset = ax.contour(xx, yy, f, colors='k')
#    ax.clabel(cset, inline=1, fontsize=10)
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    plt.title('2D Gaussian Kernel density estimation. BW=' + str(this_bw) + '. DS='+str(downsamp))
#    
#    fig.savefig('D:/data/Behavior data/RW_data/analysisOutputs/waveletDecomp/tsne_kde_BW' + str(this_bw) + '_BS'+str(downsamp) + '.png')

#%% How do I go from gaussian kde contour maps to individual data points which I can 
    #use to get traces and movies? 
#using this code for watershed:
# http://datahacker.rs/007-opencv-projects-image-segmentation-with-watershed-algorithm/

import skimage
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
#from google.colab.patches import cv2_imshow


#convert image to gray scale
fgray = (f / f.max())*255
fgray = np.uint8(fgray)
fig = plt.figure()
plt.imshow(fgray)

#distance transformation with Euclidean geom and masksize=3
dist_transform = cv2.distanceTransform(fgray, cv2.DIST_L2,3)
fig = plt.figure()
plt.imshow(dist_transform)

#find local maxima
local_max_location = peak_local_max(fgray, min_distance=1, indices=True)
local_max_boolean = peak_local_max(fgray, min_distance=1, indices=False)
print(local_max_boolean.sum())

#label markers for watershed algo as unique integers starting at 1
markers, _ = ndi.label(local_max_boolean)

#perform watershed segmentation
segmented = skimage.segmentation.watershed(255-dist_transform, markers, mask=f)



#do some plotting
fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(fgray, cmap=plt.cm.gray)
ax[0].set_title('Input image')
ax[1].imshow(-dist_transform, cmap=plt.cm.gray)
ax[1].set_title('Distance transform')
ax[2].imshow(segmented, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()




#%% Goal - take the output of the gaussian kde and use it to generate clusters. 
# Then extract points from each cluster and plot their traces. 
# will need to run kmeans first or manually label some peaks from the kde










