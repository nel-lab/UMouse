# -*- coding: utf-8 -*-
"""

to run after behavelet_and_decomp.py
 
 perfrom UMAP on a sample of the mwt data. Then embed the remaining frames 
 into that 2D space. 
 
@author: William Heffley
"""
#make a list of datasets
data_fn_list9  = list(['181215_003', '201115_000', 
                     '201217_000', '201218_000', '201226_000', '201227_000', 
                     '201228_000', '201229_000', '201230_000'])
#make a list of datasets
data_fn_list8  = list(['201115_000', 
                     '201217_000', '201218_000', '201226_000', '201227_000', 
                     '201228_000', '201229_000', '201230_000'])

#select the dataset to analyze
data_fn = data_fn_list9[2]
print(data_fn)

#set path
data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'

#%% load dependencies
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
import joblib

from datetime import date
today = date.today()
todaystr = today.strftime("%y") + today.strftime("%m") + today.strftime("%d")

#%% load spectrographic data and downsample by XX

X_new = pd.read_csv(data_dir + data_fn + '_mwtXNew.csv')
X_new = X_new.to_numpy()

downsamp = 20
XDSamp = X_new[::downsamp]

np.savetxt(data_dir + '_XDSamp' + str(downsamp) + '_' + todaystr + '.csv', 
           XDSamp,
           delimiter=",")

#%% Importance Sampling - preferentially select frames around a given event (whisker contact)

# #set interval of interest (+30:330 ms just after each whisker contact)
# int_start = 30/1000
# int_end = 330/1000

# #set stratification ratio. proportion of frames from interval of interest vs the rest of the session 
# strat_rat = 10 
# downsamp = 20

# #load spectrographic data and behavior labels
# X_new = pd.read_csv(data_dir + '_mwtXNew.csv')
# X_new = X_new.to_numpy()
# trackingDf = pd.read_csv(data_dir + '_Df.csv') 

# #determine the frame rate
# frame_rate = np.round(1/np.mean(np.diff(trackingDf['timeStamps'][0:10000])))

# #format behavior labels by retrieving first whisk contact in a sequence
# wiskArray = np.array(trackingDf['wiskContTimeBool'])
# wiskArray = np.insert(np.diff(wiskArray), 0, 0, axis=0)
# wiskArray[wiskArray == -1] = 0

# #identify frames in interval of interest
# for thisWisk in np.where(wiskArray)[0]:
#     wiskArray[thisWisk] = 0
#     wiskArray[int(thisWisk+np.round((int_start/1)*frame_rate)) : int(thisWisk+np.round((int_end/1)*frame_rate))] =1

# #retrieve spectrographic data from each period
# wiskContXNew = X_new[np.where(wiskArray[:-1])[0],:]
# otherXNew = X_new[np.where(wiskArray[:-1]==0)[0],:]
    
# #downsample from each category and combine
# #wiskContDSamp = wiskContXNew[::int(np.round(downsamp/strat_rat))]
# wiskContDSamp = wiskContXNew[::2]
# otherDSamp = otherXNew[::downsamp]

# #combine into one sample for fitting the umap embedding
# XDSamp = np.concatenate((wiskContDSamp, otherDSamp))


#%% a function which maps all data points 
def plotEmbeddedByBx(bx_labels, embedding_all, this_session, labels_included, fig_dir):
    
    # 0=other, 1=reward+500ms, 2=obst1/3,   3=obstMid,   4=obstEnd
    fig = plt.figure()
    ax = plt.axes(title='UMAP all points embedded. n='+ str(embedding_all.shape[0]) + ' ' + this_session + ' cyan=reward, gbk=obstacle')
    
    #trim because embedding has n-1 frames compared to bx_labels
    if len(bx_labels) == len(embedding_all) + 1:
        bx_labels = bx_labels[0:-1] 
    
    #plot data points
    if 'other' in labels_included:
        ax.scatter(*embedding_all.T[:,[np.where(bx_labels==0)[0]]], c='r', marker='o', s=0.2, alpha=0.01) #other
        
    if 'obst' in labels_included:
        ax.scatter(*embedding_all.T[:,[np.where(bx_labels==2)[0]]], c='g', marker='o', s=0.2, alpha=0.2) #obst1/3
        ax.scatter(*embedding_all.T[:,[np.where(bx_labels==3)[0]]], c='b', marker='o', s=0.2, alpha=0.2)
        ax.scatter(*embedding_all.T[:,[np.where(bx_labels==4)[0]]], c='k', marker='o', s=0.2, alpha=0.2)
    
    if 'reward' in labels_included:
        ax.scatter(*embedding_all.T[:,[np.where(bx_labels==1)[0]]], c='c', marker='o', s=0.2, alpha=0.2) #reward
    
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    
    if fig_dir:
        #save figure
        fig.savefig(fig_dir)
    
#%% MANY MOUSE  umap embedding

#determine # of sessions to be divided up
nn_list = [20]
md_list = [0.025]
tot_embed_fr = 50000
n_sess = len(data_fn_list8)
fr_per_sess = int(np.round(tot_embed_fr/n_sess))    

from sklearn import preprocessing

#collect a sample of frames from each session
for n_neighbors, min_dist in zip(nn_list, md_list):
    for this_sess in data_fn_list8:
        print(this_sess)
        
        #load tracking dataframe, save jaw/body angles 
        #trackingDf = pd.read_csv('D:/data/BehaviorData/RW_data/' + this_sess + '/' +this_sess + '_Df.csv')
        #trackingDf = trackingDf[['jawVarX', 'bodyAngles']].to_numpy()
        
        #load spectrographic data
        this_data_dir = 'D:/data/BehaviorData/RW_data/' + this_sess + '/' +this_sess + '_jawBod_mwtXNew.csv'
        this_spect = pd.read_csv(this_data_dir, header=None)
        this_spect = this_spect.to_numpy()
        
        # remove jaw/body angle spect data and replace with raw angles
        #this_spect = this_spect[:,0:300] 
        #this_spect = np.concatenate((this_spect, trackingDf), axis=1)
    
        #select the frames to sample
        n_frames = len(this_spect)
        samp_frames = np.round(np.linspace(0, n_frames, num=fr_per_sess, endpoint=False))
        samp_frames = list(samp_frames.astype(int))
            
        #collect sample frames 
        if this_sess == data_fn_list8[0]:
            multi_samp = this_spect[samp_frames,:]
        else:
            multi_samp = np.concatenate((multi_samp, this_spect[samp_frames,:]))
            
        del this_spect
    
    #need to use preprocessing since we are using processed and unprocessed data
    scaler = preprocessing.MinMaxScaler().fit(multi_samp)
    scaler.transform(multi_samp)
        
    #perform umap embedding
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    mapper = reducer.fit(multi_samp)
    embedding = mapper.embedding_
    
    #save the model so you can load it later
    mapper_fn = "D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/" + \
        'umap_multiMouse_model_md' + str(mapper.min_dist)[-2::] + '_nn' + str(mapper.n_neighbors) 
    joblib.dump(mapper, mapper_fn)
    
    #load the model
    # mapper = joblib.load(mapper_fn)
    
    #plot
    fig = plt.figure()
    umap.plot.points(mapper)
    plt.title('UMAP embedding: 8 mouse embedding, n=50k, min_dist=' + \
              str(mapper.min_dist) + ', n_neighbors=' + str(mapper.n_neighbors))
    fig.savefig('D:/data/BehaviorData/RW_data/analysisOutputs/multiMouse/multiMouse_md' + \
                str(mapper.min_dist)[-2::] + '_nn' + str(mapper.n_neighbors))

#%% use multimouse model to make embeddings for each mouse and save embedding

save_dir = 'D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/'    
print('make embeddings for all mice using the multimouse model')

for this_sess in data_fn_list8:
    print(this_sess)
    
    #load tracking dataframe, save jaw/body angles 
    trackingDf = pd.read_csv('D:/data/BehaviorData/RW_data/' + this_sess + '/' +this_sess + '_Df.csv')
    trackingDf = trackingDf[['jawVarX', 'bodyAngles']].to_numpy()
    
    #load spectrographic data
    this_data_dir = 'D:/data/BehaviorData/RW_data/' + this_sess + '/' + this_sess
    this_spect = pd.read_csv(this_data_dir + '_jawBod_mwtXNew.csv')
    this_spect = this_spect.to_numpy()
    # if len(trackingDf) > len(this_spect):
    #     trackingDf = trackingDf[0:-1]
    
    # remove jaw/body angle spect data and replace with raw angles
    #this_spect = this_spect[:,0:300] 
    #this_spect = np.concatenate((this_spect, trackingDf), axis=1)
        
    #need to use preprocessing since we are using processed and unprocessed data
    scaler = preprocessing.StandardScaler().fit(this_spect)
    scaler.transform(this_spect)
    
    #transform the data and save a copy
    this_embed = mapper.transform(this_spect)
    np.savetxt(this_data_dir + '_multiMouseEmbed_50k_' + todaystr + '.csv', 
            this_embed,
            delimiter=",")
    
    #plot this mouse's embedding of all data points
    bx_labels = genfromtxt(this_data_dir + '_bxLabelsArray.csv', 
                           delimiter=',')
    plotEmbeddedByBx(bx_labels, this_embed, this_sess, 
                     ['reward', 'obst', 'other'], 
                     fig_dir = this_data_dir + '_multiMouse_embedAllPoints_'+ todaystr)
    
    #save figure
    #fig.savefig(this_data_dir + '_multiMouse_embedAllPoints_' + todaystr)
    
    #clear memory 
    del this_spect, bx_labels

#%% Multi-mouse scatterplot with behavior labels 

umap_dir = 'D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/' 
save_dir = 'D:/data/BehaviorData/RW_data/analysisOutputs/multiMouse'
embed_date = '210310'
n_sess = len(data_fn_list8)
title_str = 'mulitMouse embedding n=' + str(n_sess) + ' mice'

for this_sess in data_fn_list8:
    print(this_sess)
    
    this_data_dir = 'D:/data/BehaviorData/RW_data/' + this_sess + '/' + this_sess
    
    #load the umap embedding and bheavior labels
    this_embed = genfromtxt(umap_dir + '/' + this_sess + 'multiMouseEmbed_' + embed_date + '.csv',
                            delimiter=',')
    this_bx = genfromtxt(this_data_dir + '_bxLabelsArray.csv', 
                           delimiter=',')
    this_bx = this_bx[0:-1]
    
    #select every nth frame where n=#mice
    if this_sess == data_fn_list8[0]:
        group_embed = np.array(this_embed[::n_sess])
        group_bx = np.array(this_bx[::n_sess])
    else:      
        group_embed = np.concatenate((group_embed, this_embed[::n_sess]))
        group_bx = np.concatenate((group_bx, this_bx[::n_sess]))
    
#plot all the points
plotEmbeddedByBx(group_bx, group_embed, 
                 title_str, 
                 ['reward', 'obst', 'other'],
                 fig_dir=False)

#%% SINGLE MOUSE - perform UMAP on downsampled data

reducer = umap.UMAP(n_neighbors=25, min_dist=0.05)
mapper = reducer.fit(XDSamp)
embedding = mapper.embedding_

#mapper_fn = "D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/" + 'umap_mapping_DS_' + str(downsamp) + todaystr 
#joblib.dump(mapper, mapper_fn)

#%% Visualize umap

fig = plt.figure()
umap.plot.points(mapper)
plt.title('UMAP embedding: ' + data_fn + ' DS=' + str(downsamp) + ' biased downsampling 30% post whiskCont')
#plt.title('UMAP embedding: ' + data_fn + ' DS=' + str(downsamp) + 'with jaw and body angles')

#two mouse embedding
#plt.title('UMAP embedding: ' + expt_fn1 + ' + ' + expt_fn2 + ' DS=' + str(downsamp))

#%% Map remaining data onto the embedding space

embedding_all = reducer.transform(X_new)

#embedding_fn = "D:/data/BehaviorData/RW_data/analysisOutputs/mouseLeapSavedVars/" + 'umap_mapping_all_' + todaystr 
#joblib.dump(embedding_all, embedding_fn)


#%% Plot umap points according to velocity or jaw angle

embed_date = '210310'
for this_sess in data_fn_list8:
    print(this_sess)
    
    this_data_dir = 'D:/data/BehaviorData/RW_data/' + this_sess + '/' + this_sess
    
    #load data frame and embedding
    trackingDf = pd.read_csv(this_data_dir + '_Df.csv') 
    embedding_all = genfromtxt(this_data_dir + 'multiMouseEmbed_' + embed_date + '.csv', 
                           delimiter=',')

    #get velocity variable
    # Z_Var = trackingDf['velVar'] #[:-1] #lopped off one data point because mwt is a diff transform
    # title_str = 'UMAP all points ' + this_sess + ', z=velocity, multiMouseModel'
    # vmin = -0.3
    # vmax = 1
    # plot_type = this_sess + '_multiMouseModel_velocityScatter'
    
    #get jaw variable
    Z_Var = trackingDf['jawVarY'] #[:-1] #lopped off one data point because mwt is a diff transform
    title_str = 'UMAP all points ' + this_sess + ', z=jawAngle, multiMouseModel'
    plot_type = this_sess + '_multiMouseModel_jawAngleScatter'
    
    #get body variable
    # Z_Var = trackingDf['bodyAngles'] 
    # title_str = 'UMAP all points ' + this_sess + ', z=bodyAngle, multiMouseModel'
    # plot_type = this_sess + '_multiMouseModel_bodyAngleScatter'
    
    #make scatterplot
    vmin = Z_Var.min() 
    vmax = Z_Var.max() 
    fig = plt.figure()
    
    this_scatt = plt.scatter(*embedding_all.T, 
               c=Z_Var[0:-1], 
               vmin = vmin,
               vmax = vmax,
               marker='o', s=0.5, alpha=0.2) 

    plt.title(title_str)
    plt.colorbar(this_scatt)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    fig.savefig(this_data_dir + plot_type)

#%% plot umap points according to whisker contact time

whiskCont = trackingDf['wiskContTimeBool'][:-1]

#make scatterplot 
fig = plt.figure()
ax = plt.axes(title='UMAP all points embedded. n='+ str(embedding_all.shape[0]) + ' ' + data_fn + ' blue=whiskContact, red=all other')

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
plt.title('umap gaussian kde BW=' + str(bw_val) + '. DS='+str(downsamp) + ' ' + datas_fn)

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





