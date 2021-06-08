#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:02:23 2021

@author: jimmytabet
"""

import os, cv2, h5py, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.util import montage

#%% load dataframe, can be passed as str or pandas df, or list of str/pandas df
def load_dfs(dfs):
    # convert to list
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    # iterate through each input and append to list
    dfs_all = []
    for df in dfs:
        if isinstance(df, str):
            dfs_all.append(pd.read_csv(df))
        elif isinstance(df, pd.DataFrame):
            dfs_all.append(df)
        else:
            raise ValueError('unrecognized dataframe format')
    
    return dfs_all

#%% plot embedding with behavior labels
def plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend, title='UMAP embeded points by behavior label'):
    # load dfs    
    dfs = load_dfs(dfs)
    
    fig = plt.figure()
            
    # plot 3D
    if all('dim3' in i for i in [df.columns.tolist() for df in dfs]):   
        dfs = pd.concat(dfs)

        ax = fig.add_subplot(projection='3d', title=title,
                             xlabel='dimension 1', ylabel = 'dimension 2', zlabel = 'dimension 3')
        
        for lab in np.unique(behavior_labels):
            df = dfs[behavior_labels == lab]

            ax.scatter(df['dim1'], df['dim2'], df['dim3'], label = behavior_legend[lab])
   
    # plot 2D
    else:
        dfs = pd.concat(dfs)

        ax = fig.add_subplot(title=title, xlabel='dimension 1', ylabel = 'dimension 2')
        
        for lab in np.unique(behavior_labels):
            df = dfs[behavior_labels == lab]

            ax.scatter(df['dim1'], df['dim2'], label = behavior_legend[lab])
    
    fig.legend()
    
    return fig, ax


#%% plot UMAP embedding
def plot_embedding(dfs, behavior_labels = [], behavior_legend = [], sep_data=False, title='UMAP embeded points'):
    '''

    Parameters
    ----------
    dfs : dataframe or list of dataframes
        Data to be plotted.
    sep_data : bool, optional
        Display each dataset in different color. The default is False.
    title : str, optional
        Title of UMAP plot. The default is 'UMAP embeded points'.

    Returns
    -------
    fig : matplotlib Figure
        UMAP embedding Figure.
    ax : matplotlib Axis
        UMAP embedding Axis.

    '''
    
    # plot embedding with behavior labels if given
    if len(behavior_labels):
        # raise error if legend is not given
        if not len(behavior_legend):
            raise ValueError('Provide behavior legend assoicated with behavior labels')
        
        # plot embedding with behavior labels
        fig, ax = plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend, title=title)
        return fig, ax
    
    # load dfs    
    dfs = load_dfs(dfs)
    
    # change color per data set of leave as one color
    if sep_data:
        colors = None
    else:
        colors = 'C0'
    
    fig = plt.figure()
            
    # plot 3D
    if all('dim3' in i for i in [df.columns.tolist() for df in dfs]):    
        ax = fig.add_subplot(projection='3d', title=title,
                             xlabel='dimension 1', ylabel = 'dimension 2', zlabel = 'dimension 3')
        for num, df in enumerate(dfs):
            # seperate data labels
            if sep_data:
                label = f'df_{num+1}'
            # same data labels
            else:
                label = '_'*num+'data'
            
            ax.scatter(df['dim1'], df['dim2'], df['dim3'], c = colors, label = label)
   
    # plot 2D
    else:
        ax = fig.add_subplot(title=title, xlabel='dimension 1', ylabel = 'dimension 2')
        for num, df in enumerate(dfs):
            # seperate data labels
            if sep_data:
                label = f'df_{num+1}'
            # same data labels
            else:
                label = '_'*num+'data'
            
            ax.scatter(df['dim1'], df['dim2'], c = colors, label = label)
    
    # show legend if seperate data sets
    if sep_data:
        fig.legend()
    
    return fig, ax

#%% set axes function for plotting traces
def set_axes(figure, num_clusters, num_points, show_y = False):
    subplots = figure.get_axes()
    # iterate over each cluster
    for i in range(num_clusters):
        temp_subplots = subplots[i*num_points:i*num_points+num_points]
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
    
    # set yticks (as min/max) for each point
    [ax.set_yticks(ax.get_ylim()) for ax in subplots]
    
    if not show_y:
        # remove yticks
        [ax.set_yticks([]) for ax in subplots]

#%% iteractive plotting class
class interactive():
    def __init__(self, n_clusters, n_points, spread, UMAP_dfs, behavior_labels = [], behavior_legend = []):            
        # load dfs    
        UMAP_dfs = load_dfs(UMAP_dfs)
        
        # set class variables
        self.n = n_clusters
        self.k = n_points
        self.spread = spread
        self.UMAP_dfs = UMAP_dfs
        self.behavior_labels = behavior_labels
        self.behavior_legend = behavior_legend
        
    # get points from UMAP embedding
    def get_points(self, sep_data=False):
        '''
    
        Parameters
        ----------
        n : int
            number of UMAP points to plot traces of
        dfs : dataframe or list of dataframes
            Data to be plotted.
        sep_data : bool, optional
            Display each dataset in different color. The default is False.
    
        Returns
        -------
        None.
    
        '''
        
        # set function variable to class variables
        n = self.n
        k = self.k
        UMAP_dfs = self.UMAP_dfs
        behavior_labels = self.behavior_labels
        behavior_legend = self.behavior_legend
        
        # raise error if using 3D embedding
        if all('dim3' in i for i in [df.columns.tolist() for df in UMAP_dfs]):
            raise ValueError('interactive trace plotting only available for 2D UMAP embeddings')
            
        # concat all UMAP dfs
        UMAP_dfs_all_og = pd.concat(UMAP_dfs, keys = [num for num in range(len(UMAP_dfs))])
        # only keep dim1/dim2 for plotting
        UMAP_dfs_all_og = UMAP_dfs_all_og[['dim1','dim2']]
        
        UMAP_dfs_all = UMAP_dfs_all_og.copy()
        
        # raise error if number of points > total number of frames
        if (n*k)>len(UMAP_dfs_all):
            raise ValueError(f'n ({n}) > total points ({len(UMAP_dfs_all)}), pick smaller n')
        
        # plot embedding (optionally with behavior labels)      
        fig, ax = plot_embedding(UMAP_dfs, behavior_labels = behavior_labels,
                                 behavior_legend = behavior_legend, sep_data=sep_data,
                                 title=f'UMAP embeded points\nCHOOSE {n} POINTS')
        
        # interactively select points to show traces
        selected_pts_all = []
        for i in range(n):
            # user input
            pt = plt.ginput()
            pt = np.array(pt).squeeze()
            
            # get closest k points to selected point
            dist = np.linalg.norm(UMAP_dfs_all-pt, axis=1)
            sorted_dist = np.argsort(dist)
            index = UMAP_dfs_all.iloc[sorted_dist].index
            
            # init variables to pick points that are not close or on edge of data
            idx = 0
            selected_pts = []
            idxs = []
    
            # iter through all points to find suitable ones
            for idx in index:
                # check is there are k points yet
                if len(selected_pts) == k:
                    break
                
                (df_num, frame) = idx
                skip=False
                end = len(UMAP_dfs[df_num])
                
                # check if close to previously selected point
                if len(idxs):
                    sel_pt = np.array(idxs)
                    sel_pt_df = sel_pt[sel_pt[:,0] == df_num]
                    
                    check = abs(sel_pt_df[:,1]-frame)<spread
                    skip = check.any()
                    
                # edge cases
                if frame-spread<0 or frame+spread>end-1:
                    continue
                # point close to previously selected point
                elif skip:
                    continue
                # add to list and drop to not duplicate
                else:
                    idxs.append(idx)
                    selected_pts.append(UMAP_dfs_all.loc[idx])
                    UMAP_dfs_all = UMAP_dfs_all.drop(idx)
            
            # do it again without restrictions on proximity to previous points
            if len(selected_pts) != k:
        
                # iter through all points to find suitable ones
                for idx in index:
                    if idx in idxs:
                        continue

                    # check is there are k points yet
                    if len(selected_pts) == k:
                        break
                    
                    (df_num, frame) = idx
                    end = len(UMAP_dfs[df_num])
                        
                    # edge cases
                    if frame-spread<0 or frame+spread>end-1:
                        continue
                    # add to list and drop to not duplicate
                    else:
                        selected_pts.append(UMAP_dfs_all.loc[idx])
                        UMAP_dfs_all = UMAP_dfs_all.drop(idx)
                        
            selected_pts = np.array(selected_pts)
            if len(selected_pts) != k:
                raise ValueError('Not enough eligible points. Decrease n_clus, n_points, or shorten spread')
                
            selected_pts_all.append(selected_pts)
            
            # plot k selected points with text label
            ax.scatter(*selected_pts.T, c = 'k', s=10, marker = '*', label = '_'*i+'selected\npoints')
            [plt.text(*sl_pt, str(i+1)+string.ascii_letters[j]) for j, sl_pt in enumerate(selected_pts)]
            ax.set_title(f'UMAP embeded points\nselected point(s) {i+1} of {n}')
    
            # pause to update plot with selected point
            plt.pause(1e-6)
        
        # get selected data set frame from selected points
        selected_pts_all = np.array(selected_pts_all).reshape(-1,2)
        selected_frames = [int(np.where((UMAP_dfs_all_og == pt).all(axis=1))[0]) for pt in selected_pts_all]
        
        selected_frames = np.array(selected_frames)
        
        self.selected_frames = selected_frames
        
        return selected_frames
    
    # plot traces
    def plot_traces(self, behavior_dfs, behavior_variable, selected_frames = [], show_y=False):
    
        # get interactive points from UMAP embedding
        if not len(selected_frames):
            if not hasattr(self, 'selected_frames'):
                self.get_points()

            selected_frames = self.selected_frames
        
        # set spread variable
        spread = self.spread
        
        # load dfs    
        behavior_dfs = load_dfs(behavior_dfs)
        
        # collect traces info from selected frames
    
        # concat all behavior dfs
        behavior_dfs_all = pd.concat(behavior_dfs, keys = [num for num in range(len(behavior_dfs))])
        
        # get multi-index for selceted points: data set number and frame number
        indices = behavior_dfs_all.iloc[selected_frames].index
        
        # loop through each selected point
        trace_data = []
        for pt_num, (df_num, frame) in enumerate(indices):
            # select appropriate data set
            df = behavior_dfs[df_num]
            
            trace = df.loc[frame-spread:frame+spread, behavior_variable]
    
            # add empty points if frame is near edge of data
            # frame near beginning
            if frame-spread<0:
                append = np.array([np.nan]*abs(frame-spread)*len(behavior_variable))
                append = append.reshape(-1, len(behavior_variable))
                append = pd.DataFrame(append, columns = behavior_variable,
                                      index = range(-len(append),0))
                
                trace = pd.concat([append, trace])
            
            # frame near end
            if frame+spread>len(df)-1:
                append = np.array([np.nan]*abs(frame+spread-(len(df)-1))*len(behavior_variable))
                append = append.reshape(-1, len(behavior_variable))
                append = pd.DataFrame(append, columns = behavior_variable,
                                      index = range(trace.index.max()+1,
                                                    trace.index.max()+1 + len(append)))
                
                trace = pd.concat([trace, append])
            
            # store data (data set number, frames, traces) in traces variable
            pt_dict = {'df_num': df_num+1,
                       'frame': frame,
                       'trace': trace}
            
            trace_data.append(pt_dict)
        
        # plot traces in new figure
        fig2 = plt.figure()
        fig2.set_tight_layout(True)
        fig2.suptitle('Traces of Selected Points')
    
        # loop trough trace data to plot each group of points
        for num, pt in enumerate(trace_data):
            ax = fig2.add_subplot(n, k, num+1)
            ax.plot(pt['trace'])
            if len(behavior_dfs) > 1:
                ax.set_title(f'Point {num//k+1}{string.ascii_letters[num%k]}, df_{pt["df_num"]}')
            else:
                ax.set_title(f'Point {num//k+1}{string.ascii_letters[num%k]}')
            
            # show beginning, middle, and end frames            
            ax.set_xticks([pt['trace'].index[pt['trace'].index>=0][0],
                           pt['frame'],
                           pt['trace'].index[pt['trace'].notnull().all(1)].max()])
            
            # # only show middle frame
            # ax.set_xticks([pt['frame']])
            
            # show entire spread even if near edge
            ax.set_xlim(pt['frame']-spread, pt['frame']+spread)
            
        # make ylim the same for each cluster of points
        set_axes(fig2, n, k, show_y=show_y)
        
        fig2.legend(behavior_variable)
        
    # behavior montage
    def behavior_montage(self, video_path, save_path, fps, indices = []):
        
        '''
    
        Parameters
        ----------
        video_path : str
            Path to behavior video. Must be hdf5.
        save_path : str
            Path to save behavior montage. Must include extension/format (.avi, .mp4, .mov).
        indices : numpy array
            Array of selected frame incides.
        spread : int
            Number of frames to show on either side of indexed frame.
        fps : int, optional
            Frame rate of the behavior video. The default is 70.
    
        Returns
        -------
        None. Saves behavior montage to save_path.
    
        '''
        
        # get interactive points from UMAP embedding
        if not len(indices):
            if not hasattr(self, 'selected_frames'):
                self.get_points()

            indices = self.selected_frames
                
        # set function variable to class variables
        n = self.n
        k = self.k
        spread = self.spread
        
        # raise error if save_path does not include extension
        if len(os.path.splitext(save_path)[-1]) == 0:
            raise ValueError('Specify format/extension for save_path')
                
        # get sorted indices (needed for h5py indexing)
        indices_sorted = np.sort(indices)
        
        # reset indices from sorted
        reset_indices = indices_sorted.searchsorted(indices)
        
        # init big movie
        big_mov = []
        
        # open hdf5 movie
        with h5py.File(video_path, 'r') as hf:
            # get key (assuming first key)
            key = [key for key in hf][0]
    
            # save montage of points for each frame
            for i in range(2*spread+1):
                # index using sorted indices
                frames_sorted = hf[key][indices_sorted-spread+i]
                # use original frame order using reset indices
                frames = frames_sorted[reset_indices]
                big_frame  = montage(frames, grid_shape=(n,k))
                big_mov.append(big_frame)
        
        # init montage video
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(save_path, fourcc, fps, big_frame.shape[::-1])
        
        # save each frame (3 channels needed) to montage
        for frame in big_mov:
            video.write(cv2.merge([frame]*3))
         
        video.release()
        print(f'behavior montage saved to {save_path}')
        
        # # play video - can't set frame rate?
        # cap = cv2.VideoCapture(save_path)
        
        # # Check if camera opened successfully
        # if (cap.isOpened()== False): 
        #     raise ValueError('Error opening video file')
        
        # # Read until video is completed
        # while(cap.isOpened()):
        #     # Capture frame-by-frame
        #     ret, frame = cap.read()
        #     if ret == True:
          
        #         # Display the resulting frame
        #         cv2.imshow('Frame',frame)
            
        #         # Press Q on keyboard to  exit
        #         if cv2.waitKey(25) & 0xFF == ord('q'):
        #             break
          
        #     # Break the loop
        #     else: 
        #       break
        
        # # When everything done, release the video capture object
        # cap.release()
        
        # # Closes all the frames
        # cv2.destroyAllWindows()

#%% random test example
a = np.random.rand(10,3)
b = np.random.rand(10,3)
c = np.random.rand(10,3)
a_df = pd.DataFrame(a, columns = ['dim1','dim2','dim9'])
b_df = pd.DataFrame(b, columns = ['dim1','dim2','dim9'])
c_df = pd.DataFrame(c, columns = ['dim1','dim2','dim9'])

# UMAP_dfs = [a_df]
UMAP_dfs = [a_df, b_df, c_df]

UMAP_dfs_all = pd.concat(UMAP_dfs)
behavior_labels = np.random.randint(0,4,len(UMAP_dfs_all))
behavior_legend = [f'beh_{num}' for num in range(max(np.unique(behavior_labels))+1)]
UMAP_dfs_all['label']=behavior_labels

# plot_embedding(UMAP_dfs, behavior_labels, behavior_legend)
# interactive_plot_traces(3, 5, UMAP_dfs, UMAP_dfs, ['dim9','dim2'], 3, behavior_labels, behavior_legend)

# print(UMAP_dfs_all.iloc[pts.flatten()])

#%%
UMAP_dfs = pd.read_csv('/Users/jimmytabet/Desktop/plotting_UMouse demo/nmf.csv')
behavior_dfs = '/Users/jimmytabet/Desktop/plotting_UMouse demo/behavior.csv'
behavior_variable = ['Lpaw_Y', 'Rpaw_Y']
n = 3
k = 5
spread=70
fps = 70
video_path = '/Users/jimmytabet/NEL/Projects/BH3D/mov.h5'
save_path = '/Users/jimmytabet/Desktop/test.avi'

inter = interactive(n,k,spread,UMAP_dfs)
inter.get_points(sep_data=True)
inter.plot_traces(behavior_dfs, behavior_variable)
inter.behavior_montage(video_path, save_path, fps)