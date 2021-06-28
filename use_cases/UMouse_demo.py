#!/usr/bin/env python
# coding: utf-8

"""
@author: JakeHeffley

UMouse demo file for fitting a UMAP embedding and visualizing the results. 
It is best run via the Spyder IDE with the matplotlib back end set to:

%matplotlib

Drop box link for demo behavior dataframes
https://www.dropbox.com/sh/sn1ru8sf19icb4u/AAA7Q70qVq2XwVSMywmG0FOpa?dl=0
*NOTE: Only the .csv files are required for most of the demo. The .mp4 files 
are only requied to utilize the behavior_montage method of the interactive 
class object.

Download the two dataframes to a local directory. 

The dropbox folder contains two pandas dataframes with 3D positional coordinates for all 
four paws. They also contain boolean variables with 1s representing frame indeces
of trial/behavioral events such as the obstacle periods, reward delivery, and 
licking.

The demo will take the xyz positional coordinates for each paw and transform them 
using the Morlet wavelet transform. Next, a subset of the wavelet data will be 
used to fit a 2 dimensional UMAP embedding. For the purposes of this demo both
the number of frames used to fit the embedding and the total number of frames 
have been limited. 

The demo will save a file for each dataset with the tags _mwt for the output 
of the morlet wavelet transformation. An additional saved file with the tag 
_umap will contain the corrdinates in UMAP embedding space for each point which
is transformed. These files will be saved into the same directory as the input
data. 
 
"""

#%% import

# Add UMouse repo to PYTHONPATH!
# if using Spyder, click on Python logo in top bar to add to PYTHONPATH
# if using Jupyter Notebook:
'''
import sys
sys.path.append('path/to/UMouse')
'''

from umouse.UMouse import UMouse
import umouse.UMousePlotter as UMPlot
import pandas as pd
import numpy as np

#%% Set pathname to the folder containing the two demo dataframes



data_path = 'local/path/to/dataframes/folder/'




#%% generate mwt and UMAP embedding for mice

#initialize the UMouse class object and set parameters for the morlet wavelet transform and UMAP embedding
um_estimate = UMouse(n_frequencies=25, f_sample=250, fmin=1, fmax=None, n_neighbors=15, n_components=2)

# instruct UMouse to use specific columns of the data frame for the embedding. 
# in this case columns refer to all four paws (Front/Back Left/Right) and 3 axes for each paw (X,Y,Z)
columns_list = 'BLX', 'BLY', 'BLZ', 'FLX', 'FLY', 'FLZ', 'FRX', 'FRY', 'FRZ', 'BRX', 'BRY', 'BRZ'

#set first argument equal to list of filenames for datasets to be analyzed
df = [data_path +'201226_000_behavior_df.csv',
      data_path + '201227_000_behavior_df.csv'
      ]

#use 10000 frames from each dataset to fit the embedding model
fit_data = um_estimate.fit(df, fr_per_sess=10000, columns=columns_list)

#%% transform data 

#This step may take 2-6 minutes depending on your compute power. 

#pass a list of filenames to the transform method. It will automatically look for versions of those files tagged with "_mwt"
um_estimate.transform(df)

#%% use UMouse_Plotter_functions to visualize the UMAP embedding in 2 dimensions

#set paths for umap embedding files
embedding_paths = [this_df.split('.')[0] + '_umap.csv' for this_df in df]

#%% plot UMAP embedding. Each frame/timepoint in the transformed dataset will be 
# represented by a point in the 2 dimnesional umap embedding space. Timing of 
# relevant trial/behavioral events can be indicated using color axis. 

#use behavior dataframe to create frame indeces and a legend for behavior events
#this will be unique to your data. For the demo datasets the categories are:

# 0 = all unlabeled 
# 1 = a window just after reward delivery
# 2 = periods in which the animal is jumping over a moving obstacle
 
behavior_legend = ['other', 'reward', 'obstacle']
behav_df = pd.read_csv(df[0], usecols=['rewardBool', 'obstacleBool'])
behavior_labels = behav_df['rewardBool'].to_numpy(dtype='int')
behavior_labels[np.where(behav_df['obstacleBool']==1)] = 2

UMPlot.plot_categorical_vars(embedding_paths[0], behavior_labels, behavior_legend, ds=1, save=False,
                                      s=0.4, alpha=0.5)

#%% Quiver plot will show how the points move through the umap embedding space 
# with respect to time. For each frame an arrow is plotted which shows the direction
# of the location of the subsequent frame's location in embedding space.
# if down_samp >1 then arrows will show the location of the next frame included
# post down sampling. 
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#plot a single dataset for the quiver plot
embedding = pd.read_csv(embedding_paths[0], index_col=0)

#down sample by 25x to improve plot visibility
UMPlot.vector_field_plot(embedding, down_samp=1, z_axis='direction')

#%% plot gradient plot for velocity 

#load velocity variable within the behavior data frame and convert to np array
velocity_arr = pd.read_csv(df[0], usecols=['velVar']).to_numpy()

#color code UMAP embedding points according to velocity 
UMPlot.plot_continuous_var(embedding, z_var=velocity_arr, ds=1, title_str='locomotion velocity')

#%% Interactive plots 

#%matplotlib

df_path = df[0]
umap_path = embedding_paths[0]
video_path = data_path + '201226_000_run.mp4'
save_path = data_path + '201226_000_montage.avi'

fps = 250

#initialize the interactive() class object
um_int = UMPlot.interactive(n_clusters=3, k_neighbors=3, spread=250, UMAP_dfs=umap_path, ds=20)

#select 3 data from the UMAP embedding
# get the three nearest data points around selected point 
um_int.get_points()

#%% plot traces (using UMAP data as behavior data) 
um_int.plot_traces(df_path, ['FLX', 'FLY', 'FLZ'])


#%% Create montage of videos for windows around the selected points. +/- spread frames
um_int.behavior_montage(video_path, save_path, fps) 

#press q to stop the movie
um_int.play(save_path, loop=False, title='Behavior movie for selected points')









