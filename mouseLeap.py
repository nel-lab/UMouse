# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:19:27 2021

to run after behavelet_and_decomp.py
 
 perfrom UMAP on a sample of the mwt data. Then embed the remaining frames 
 into that 2D space. 
 
@author: Jake
"""
#%% get started

#select the dataset to analyze
data_fn = 'trackingData_201115_000'
#data_fn = 'trackingData_181215_003'
expt_fn = data_fn[-10:]

#set path
data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'

#load dependencies
import pandas as pd
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans, DBSCAN
#from sklearn.mixture import GaussianMixture
import datashader as ds
import colorcet as cc
import umap
import umap.plot
from scipy.stats import gaussian_kde
import cv2
from datetime import date
import joblib

today = date.today()
todaystr = today.strftime("%y") + today.strftime("%m") + today.strftime("%d")

#%% load spectrographic data and downsample by XX

X_new = pd.read_csv(data_dir + expt_fn + '_mwtXNew.csv')
X_new = X_new.to_numpy()

downsamp = 20
XDSamp = X_new[::downsamp]

np.savetxt(data_dir + expt_fn + '_XDSamp' + str(downsamp) + '_' + todaystr + '.csv', 
           XDSamp,
           delimiter=",")
#np.savetxt("D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/XSDamp" + str(downsamp) + todaystr + '.csv', XDSamp, delimiter=",")


#%% TWO MOUSE EMBEDDING
#load spectrographic data for two mice and combine into one down sampled array

#select the dataset to analyze
data_fn1 = 'trackingData_201115_000'
expt_fn1 = data_fn1[-10:]
data_fn2 = 'trackingData_181215_003'
expt_fn2 = data_fn2[-10:]

#set paths
data_dir1 = 'D:/data/BehaviorData/RW_data/' + data_fn1 + '/'
#data_dir2 = 'D:/data/BehaviorData/RW_data/' + data_fn2 + '/'
data_dir2 = 'D:/data/BehaviorData/RW_data/X_newArray_181215_003'

#load spectrographic data
X_new1 = pd.read_csv(data_dir1 + expt_fn1 + '_mwtXNew.csv')
#X_new2 = pd.read_csv(data_dir2 + expt_fn2 + '_mwtXNew.csv')
X_new2 = pd.read_csv(data_dir2 + '.csv')

#downsample and combine data
downsamp = 40
XDSamp1 = X_new1[::downsamp]
XDSamp2 = X_new2[::downsamp]
XDSampBoth = np.concatenate((XDSamp1, XDSamp2))
print('XDSampBoth shape = ' + str(XDSampBoth.shape))
print('XDSamp1 shape = ' + str(XDSamp1.shape))
print('XDSamp2 shape = ' + str(XDSamp2.shape))

#perform umap embedding
reducer = umap.UMAP(n_neighbors=25, min_dist=0.05)
mapper = reducer.fit(XDSampBoth)
embedding = mapper.embedding_


#%% SINGLE MOUSE - perform UMAP on downsampled data

reducer = umap.UMAP(n_neighbors=25, min_dist=0.05)
mapper = reducer.fit(XDSamp)
embedding = mapper.embedding_

#mapper_fn = "D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/" + 'umap_mapping_DS_' + str(downsamp) + todaystr 
#joblib.dump(mapper, mapper_fn)

#%% Visualize umap

fig = plt.figure()
umap.plot.points(mapper)
plt.title('UMAP embedding: ' + expt_fn + 'DS=' + str(downsamp) + 'with jaw and body angles')

#two mouse embedding
#plt.title('UMAP embedding: ' + expt_fn1 + ' + ' + expt_fn2 + ' DS=' + str(downsamp))

#%% Map remaining data onto the embedding space

embedding_all = reducer.transform(X_new)

#two-mouse embedding
#embedding_all = reducer.transform(np.concatenate((X_new1, X_new2)))

#embedding_fn = "D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/" + 'umap_mapping_all_' + todaystr 
#joblib.dump(embedding_all, embedding_fn)

#%% Map all embedded points as 2D scatter plot
 
#load labels 
PC_labels = genfromtxt(data_dir + expt_fn + '_bxLabelsArray.csv', delimiter=',') #'D:/data/BehaviorData/RW_data/trackingArray.csv', delimiter=','
PC_labels = PC_labels[:-1]

# 0=other, 1=reward+500ms, 2=obst1/3,   3=obstMid,   4=obstEnd
fig = plt.figure()
ax = plt.axes(title='UMAP all points embedded. n='+ str(embedding_all.shape[0]) + ' ' + expt_fn + ' cyan=reward, black=late obstacle')

# ax.scatter(*embedding_all.T, s=0.1, alpha=0.05)

# ax.scatter(*embedding_all.T[:,[np.where(PC_labels==0)[1]]], c='r', marker='o', alpha=0.05) #other

ax.scatter(*embedding_all.T[:,[np.where(PC_labels==2)[0]]], c='g', marker='o', s=0.5, alpha=0.2) #obst1/3
ax.scatter(*embedding_all.T[:,[np.where(PC_labels==3)[0]]], c='b', marker='o', s=0.5, alpha=0.2)
ax.scatter(*embedding_all.T[:,[np.where(PC_labels==4)[0]]], c='k', marker='o', s=0.5, alpha=0.2)
ax.scatter(*embedding_all.T[:,[np.where(PC_labels==1)[0]]], c='c', marker='o', s=0.5, alpha=0.2) #reward

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')


#%% Plot umap points according to velocity

#two mouse embedding

# #load data and combine booleans
# trackingDf = pd.read_csv(data_dir + expt_fn + '_Df.csv') 
# velVar1 = trackingDf['velVar'][:-1] 
# data_dir2 = 'D:/data/BehaviorData/RW_data/trackingData_181215_003/'
# trackingDf = pd.read_csv(data_dir2 + expt_fn2 + '_Df.csv')
# velVar2 = trackingDf['velVar'][:-2]
# velVar = np.concatenate((velVar1, velVar2))

#load data frame
trackingDf = pd.read_csv(data_dir + expt_fn + '_Df.csv') 

#get velocity variable
velVar = trackingDf['velVar'] #[:-1] #lopped off one data point because mwt is a diff transform

#make scatterplot
fig = plt.figure()

vel_scatt = plt.scatter(*embedding_all.T, 
           c=velVar, 
           vmin = -0.3,
           vmax = 1,
           marker='o', s=0.5, alpha=0.2) 
#plt.title('UMAP all points embedded. n='+ str(embedding_all.shape[0]) + ' ' + expt_fn + ' color = velocity')
#plt.title('UMAP all points embedded. n='+ str(embedding_all.shape[0]) + ' ' + expt_fn + ' ' + expt_fn2 +' color = velocity')
plt.title('UMAP all points 201115, body and jaw included.')
plt.colorbar(vel_scatt)
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')



#%% plot umap points according to whisker contact time

whiskCont = trackingDf['wiskContTimeBool'][:-1]

#make scatterplot 
fig = plt.figure()
ax = plt.axes(title='UMAP all points embedded. n='+ str(embedding_all.shape[0]) + ' ' + expt_fn + ' blue=whiskContact, red=all other')

ax.scatter(*embedding_all.T[:,[np.where(whiskCont==0)[0]]], c='r', marker='o', s=0.5, alpha=0.2) #obst1/3
ax.scatter(*embedding_all.T[:,[np.where(whiskCont==1)[0]]], c='b', marker='o', s=0.5, alpha=0.2)

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')

#%% Calculate the Gaussian KDE

dim_red_method = "UMAP"

if isinstance(embedding, pd.DataFrame):
    embedding_df = embedding
else:
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
bw_val = 0.15
kde_kernel = gaussian_kde(values, bw_method = bw_val)
f = np.reshape(kde_kernel(positions).T, xx.shape)

fig = plt.figure()
plt.imshow(f)
plt.title('umap gaussian kde BW=' + str(bw_val) + '. DS='+str(downsamp) + ' ' + expt_fn)

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
plt.title(dim_red_method + ': 2D Gaussian KDE. BW=' + str(bw_val) + '. DS='+str(downsamp))

#%% openCV version of watershed
# https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

#convert image to gray scale
fgray = (f / f.max())*255
fgray = np.uint8(fgray)
fig = plt.figure()
plt.imshow(fgray)

#threshold the image
ret, thresh = cv2.threshold(fgray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# For some reason the image is inverted with foreground = 0. Change that
thresh = -thresh+255
fig=plt.figure()
plt.imshow(thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
fig=plt.figure()
plt.imshow(opening)
plt.colorbar()

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
plt.imshow(sure_bg)
plt.colorbar()
plt.title('sure background: 0=bg')

# Distance transform
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform)
plt.title('distance transform')

# Finding sure foreground area
#ret2, sure_fg2 = cv2.threshold(dist_transform,0.55*dist_transform.max(),255,0)
ret, sure_fg = cv2.threshold(fgray,0.75*fgray.max(),255,0)
fig = plt.figure()
plt.imshow(sure_fg)
plt.title('sure foreground')

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
fig =plt.figure()
plt.imshow(unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# Perform water shed and draw markers onto f
watershed_out = cv2.watershed(np.dstack([fgray]*3), markers)
f[watershed_out == -1] = [255]

# Plot marked regions
fig = plt.figure() 
plt.imshow(watershed_out)
plt.title('watershed_out')

#Plot gaussian output 
fig = plt.figure() 
plt.imshow(f)

plt.imshow(markers)
plt.title("markers plot")

#%% Use plt.ginput to get specific points from the scatter plot
from sklearn.preprocessing import minmax_scale

n_pts = 5  #number of points to select from the plot

iirow, iicol = np.where(watershed_out == -1)

#normalize border values
iicol = minmax_scale(iicol, feature_range=(ymin,ymax))
iirow = minmax_scale(iirow, feature_range=(xmin,xmax))

#plot watershed boundaries on top of gaussian
%matplotlib qt
fig = plt.figure()
plt.imshow(np.rot90(fgray), extent=[xmin, xmax, ymin, ymax])
plt.scatter(iirow, iicol, c='k', s=0.3)
plt.title('umap kde: select ' + str(n_pts) + ' points with the mouse')

#use ginput to select data points to sample
pts = np.asarray(plt.ginput(n_pts, timeout=-1))

#%% kclosest points to centroids function

def kclosest(k, near, dim_red_data, cluster_centroids):
    
    # k - # of points to take 
    # near - larger sample of points around centroid from which to randomly select k
    # dim_red - embedded data points after dimensionality reduction 
    # cluster centroids - centroid locations or input selected by user
    
    #dist - 
    #closeest - 
    #ids - 
    
    # get random ids of points k points from near closest points
    id_near = np.sort(np.random.choice(near, k, replace=False))
    
    # store all distances, closest ids, and k random ids from near closest
    dist = []
    closest = []
    ids = []
    #for each selected data point, 
    for centroid in zip(cluster_centroids):
        
        # calculate euclidean distance from all data points
        diff = dim_red_data-centroid
        temp_dist = np.linalg.norm(diff,axis=1)
        temp_dist = np.array(temp_dist)
        
        #dist.append(temp_dist)  #just silencing this value to reduce run time
        temp_closest = np.argsort(temp_dist, axis=0) #gives frame # sorted by distance from centroid
        closest.append(temp_closest[id_near]) #frame #s for selected k datapoints
        ids.append(temp_closest[id_near])
        dist.append(temp_dist[temp_closest[id_near]]) #distances for selected k datapoints
        
        #watershed outputs have been binned!!!!
        #cannot select arbitrary coords from the watershed. 
        
    return dist, closest, ids

#%% k random points from n_points nearest points

k = 5
n_points = 25
umap_dist, umap_closest, umap_ids = kclosest(k, n_points, embedding_all, pts)

#%% plot the selected points on top of the watershed borders

fig = plt.figure() 
plt.imshow(np.rot90(watershed_out), extent=[xmin, xmax, ymin, ymax])
plt.scatter(x = pts[:,0], y=pts[:,1], c='r', s=5)

plt.title('watershed_out with selected_points')

#%% show k closest points

# show comparison
plt.imshow(np.rot90(fgray), extent=[xmin, xmax, ymin, ymax])      
for i_dim in zip(umap_ids):
    plt.scatter(embedding_all[[i_dim[0]],[0]], embedding_all[[i_dim[0]],[1]], c='k', marker='*')

#%% subplot zoom function 
#   allows you to zoom in on a particular subplot using shift+left click
#   from https://stackoverflow.com/questions/44997029/matplotlib-show-single-graph-out-of-object-of-subplots
def add_subplot_zoom(figure):

    zoomed_axes = [None]
    def on_click(event):
        ax = event.inaxes

        if ax is None:
            # occurs when a region not in an axis is clicked...
            return

        # we want to allow other navigation modes as well. Only act in case
        # shift was pressed and the correct mouse button was used
        if event.key != 'shift' or event.button != 1:
            return

        if zoomed_axes[0] is None:
            # not zoomed so far. Perform zoom

            # store the original position of the axes
            zoomed_axes[0] = (ax, ax.get_position())
            ax.set_position([0.1, 0.1, 0.85, 0.85])

            # hide all the other axes...
            for axis in event.canvas.figure.axes:
                if axis is not ax:
                    axis.set_visible(False)

        else:
            # restore the original state

            zoomed_axes[0][0].set_position(zoomed_axes[0][1])
            zoomed_axes[0] = None

            # make other axes visible again
            for axis in event.canvas.figure.axes:
                axis.set_visible(True)

        # redraw to make changes visible.
        event.canvas.draw()

    figure.canvas.mpl_connect('button_press_event', on_click)
    
#%% set axes function
def set_axes(figure, n_rows, n_cols, show_y = False):
    subplots = figure.get_axes()
    # iterate over each cluster
    for i in range(n_rows):
        temp_subplots = subplots[i*n_cols:i*n_cols+n_cols-1]
        # initialize min/max_ylim
        min_ylim, max_ylim = np.inf, -np.inf

        # iterate over each point in each cluster
        for j in temp_subplots:
            # find min/max_ylim
            temp_min_ylim, temp_max_ylim = j.get_ylim()
            min_ylim = min(min_ylim, temp_min_ylim)
            max_ylim = max(max_ylim, temp_max_ylim)
        
        # set constant ylim for each cluster
        [ax.set_ylim([min_ylim, max_ylim]) for ax in temp_subplots]
        # "sharey" (hide yaxis for all but first point)
        [ax.yaxis.set_visible(False) for ax in temp_subplots[1:]]
        
    # set xticks (middle frame) for each point
    [ax.set_xticks([int(np.median(ax.get_xticks()))]) for ax in subplots]
    
    # set yticks (as min/max) for each point
    [ax.set_yticks(ax.get_ylim()) for ax in subplots]
    
    if not show_y:
        # remove yticks
        [ax.set_yticks([]) for ax in subplots]
        
#%% load positional data

from numpy import genfromtxt
trackingDf = pd.read_csv('D:/data/BehaviorData/RW_data/trackingDf.csv')
trackingDf.drop(labels=[0,len(trackingDf)-1], inplace=True) #remove two frames so it matches the indexing for dim reduction

#%% show traces from each group
# plot traces around kclosest points - paw center 'Y'
spread=70   # plot k closest frames (+- spread)
paws = [col for col in trackingDf.columns if col[2] == 'X']


fig = plt.figure()
fig.set_tight_layout(True)
fig.suptitle('DLC traces plotted from UMAP embedding points', size='x-large')

for this_pt in range(n_pts):
    for this_k in range(k+1):
        ax = fig.add_subplot(n_pts,k+1, (k+1)*this_pt+this_k+1)
        if this_k==k:
            ax.imshow(np.rot90(fgray), extent=[xmin, xmax, ymin, ymax])
            ax.scatter(iirow, iicol, c='k', s=0.3)
            i_dim = umap_ids[this_pt]
            ax.scatter(embedding_all[[i_dim[0]],[0]], embedding_all[[i_dim[0]],[1]], c='k', marker='*')
            ax.set_title('selected points')
        else:
            ax.plot(trackingDf.loc[umap_ids[this_pt][this_k]-spread:umap_ids[this_pt][this_k]+spread+1,paws])
        if this_k==0:
            ax.set_ylabel('selected point #'+str(this_pt+1))
        if this_pt==0 and this_k < k:
            ax.set_title('closest: '+str(this_k+1))
        if this_k == 0 and this_pt == 0:
            ax.legend(paws)

# set axes and figsize
set_axes(fig,n_pts,k+1)
DPI = fig.get_dpi()
fig.set_size_inches(1350.0/float(DPI),750.0/float(DPI))
add_subplot_zoom(fig)

#fig.savefig()

#%% Plot videos from each cluster in order to label them
#%% implement behavior montage
from use_cases.behavior_montage import behavior_montage

raw_video = 'D:/data/BehaviorData/RW_data/181215_003 .mp4'

# if hasattr(behavior_montage, 'mov'): del behavior_montage.mov

mont = behavior_montage(raw_video, downsample*pca['kmeans_ids'], shrink_factor=3, spread=spread)
mont.fr = 70

#%%
mont.save('/Users/jimmytabet/Desktop/pca_kmeans.avi')

#%% Generate a labelled behavior map in 2D 





#%% End of currently used code. Below is snippets previously used or unused.
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

#%% Perform clustering on spectrographic sample

# #kmeans clustering
# clus=10  #arbitrarily picking 15 clusters
# Cluster = KMeans(n_clusters = clus)
# kmeans = Cluster.fit(XDSamp)
# y_kmeans = kmeans.labels_ #tsneKmeans.predict(tsne)

# #Test alternate clustering methods
# #y_dbscan = DBSCAN(eps=2, min_samples=5).fit(XDSamp)
# #y_gm = GaussianMixture(n_components=2, random_state=0).fit(XDSamp)

# #identfy the number of data points in each cluster
# fig = plt.figure() 
# plt.hist(y_kmeans)
# plt.title('num clusters = ' + str(clus))

#%% Sample uniformly from each cluster

# #find min number of samples in smallest cluster 
# minbin = plt.hist(y_kmeans)[0].min()

# #identify indeces for each cluster within XDSamp/y_kmeans
# for this_clus in np.unique(y_kmeans):
#     this_clus_ind = np.where(y_kmeans == this_clus)[0]
#     #extract random indeces of samples from a given cluster
#     this_sub_samp = np.random.choice(this_clus_ind.astype(int), size=minbin.astype(int), replace=False)
#     #concatenate random samples into a new array 
#     if this_clus ==0 :
#         sub_samp = this_sub_samp
#     else:
#         sub_samp = np.append(sub_samp, this_sub_samp)

# #randomize order so clusters are not grouped
# np.random.shuffle(sub_samp)

#%% Use samples from clusters as input to umap to generate a 2D embedding

#comp = 2
# t = TSNE(n_components=comp, verbose = 2)
# tsne = t.fit_transform(XDSamp.iloc[sub_samp,:])
# np.savetxt("D:/data/BehaviorData/RW_data/tsneOutDS" + str(downsamp) + "_kmeans_subsamp.csv", tsne, delimiter=",")
