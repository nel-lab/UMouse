# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:55:13 2021

use gif package to plot individual trial trajectories through the umap embedding as 
a function of time. 

@author: Jake
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from numpy import genfromtxt
from scipy.io import loadmat

from scipy import interpolate
import math

#make a list of datasets
data_fn_list  = list(['201115_000', 
                     '201217_000', '201218_000', '201226_000', '201227_000', 
                     '201228_000', '201229_000', '201230_000'])

#select the dataset to analyze
data_fn = data_fn_list[2]
print(data_fn)

#set path
data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'
embed_date = '210331'

#%% define the plotting function

def plot_trial_trajectory(rewardFrames, obstFrames, umap_embedding, plotting_timestep, plot_n_trials):
    
    #set trial start at 0 or prev reward +1.5s 
    trial_starts = np.append(0, rewardFrames[0:-1]+375)
    trial_ends = rewardFrames + 375 #1.5s
    
    #obstacle variables
    obst_starts = obstFrames[:,0]
    obst_ends = obstFrames[:,1]
    
    cmap = mpl.cm.plasma

    #make loops for each group of four trials 
    for subplot_ix,this_trial in enumerate(range(1,plot_n_trials+1)):
        
        if subplot_ix == 0 or subplot_ix % 4 ==0:
            fig = plt.figure(tight_layout=True)
            fig.suptitle(data_fn + ': umap space trajectories. Trials ' + str(this_trial) + ':' + str(this_trial+3))
            ax = plt.subplot(2,2, 1 + subplot_ix % 4)
        else:
            ax = plt.subplot(2,2, 1 + subplot_ix % 4, sharex=ax, sharey=ax)
        
        #get total number of frames in the trial and downsample for visibility in plotting
        num_frames =  trial_ends[this_trial] - trial_starts[this_trial]
        sparse_frames = np.round(np.linspace(trial_starts[this_trial], 
                                             trial_ends[this_trial], 
                                             num=int(np.round(num_frames/plotting_timestep))))
        
        for sparse_ix in range(1, len(sparse_frames)-1):
            this_frame = int(sparse_frames[sparse_ix])
            next_frame = int(sparse_frames[sparse_ix+1])
            this_x1, this_y1 = umap_embedding[this_frame]
            this_dx, this_dy = umap_embedding[next_frame,:] - umap_embedding[this_frame,:]
            
            #plot an arrow for each timestep
            ax.arrow(x=this_x1, y=this_y1, 
                      dx=this_dx, dy=this_dy, 
                      length_includes_head=True,
                      color=cmap(sparse_ix/(len(sparse_frames)-1)),
                      head_width = 0.1)
        
        #plot points for obstacle start and stop
        this_obst_starts = obst_starts[(obst_starts>trial_starts[this_trial]) & (obst_starts<trial_ends[this_trial])]
        this_obst_ends = obst_ends[(obst_ends>trial_starts[this_trial]) & (obst_ends<trial_ends[this_trial])]
        
        ax.plot(umap_embedding[this_obst_starts,0], 
                 umap_embedding[this_obst_starts,1], 
                 color='g', 
                 marker = 'o',
                 linestyle = 'none'
                 )
        ax.plot(umap_embedding[this_obst_ends,0], 
                 umap_embedding[this_obst_ends,1], 
                 color='r', 
                 marker = 'o',
                 linestyle = 'none'
                 )
        
        for ax in fig.get_axes():
            ax.label_outer()
            
#%% import umap embedding points and the timestamps from the dataframe

#get frame numbers for rewardTimes and obstacleTimes
trackingDf = pd.read_csv(data_dir + data_fn + '_Df.csv')
rewardFrames = trackingDf['rewardBool']
rewardFrames = np.where(np.diff(rewardFrames)==1)[0] + 1
obstFrames = trackingDf['obstacleBool']
obstFrames = np.stack((np.where(np.diff(obstFrames)==1)[0] + 1, np.where(np.diff(obstFrames)==-1)[0] + +1), 
                      axis=1)

#import data for embedded point locations
umap_embedding = genfromtxt(data_dir + data_fn + '_multiMouseEmbed_50k_' + embed_date + '.csv', 
                           delimiter=',')

plotting_timestep = 10 #40ms
plot_n_trials = 12
plot_trial_trajectory(rewardFrames, obstFrames, umap_embedding, plotting_timestep, plot_n_trials)


#%% Force field plotting 

down_samp = 100
cmap = mpl.cm.hsv #cyclical cmap for angle

#load data points 
# umap_embedding = genfromtxt(data_dir + data_fn + '_multiMouseEmbed_50k_' + embed_date + '.csv', 
#                            delimiter=',')

#downsample and calculate vector components
x = umap_embedding[0::down_samp,0]
y = umap_embedding[0::down_samp,1]
u = x[1::] - x[0:-1]
v = y[1::] - y[0:-1]
 
# use quiver to plot the arrows
plt.figure(1, dpi=200)
plt.quiver(x[0:-1], y[0:-1], u, v)

#calculate anlge of each arrow
theta = math.atan(u/v)
theta *= 180/3.1415 # rads to degs


# use interpolation to creat a 2D field of vectors
xx_min, yy_min = np.min(umap_embedding,axis=0)
xx_max, yy_max = np.max(umap_embedding,axis=0)

xx = np.linspace(np.floor(xx_min), np.ceil(xx_max), 100)
yy = np.linspace(np.floor(yy_min), np.ceil(yy_max), 100)
xx, yy = np.meshgrid(xx, yy)  

points = np.transpose(np.vstack((x[0:-1], y[0:-1])))
u_interp = interpolate.griddata(points, u, (xx, yy), method='linear')
v_interp = interpolate.griddata(points, v, (xx, yy), method='linear')
#xi=points at which to interpolate data

plt.figure(2, dpi=200)
plt.quiver(xx, yy, u_interp, v_interp)
plt.show()






