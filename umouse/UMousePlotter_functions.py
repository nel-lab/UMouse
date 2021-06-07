#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:02:23 2021

@author: jimmytabet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

#%% plot embedding with behavior labels
def plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend, title='UMAP embeded points by behavior label'):
    
    # convert df to list
    if not isinstance(dfs, list):
        dfs = [dfs]
    
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
        # raise error if legend isnot given
        if not len(behavior_legend):
            raise ValueError('Provide behavior legend assoicated with behavior labels')
        
        # plot embedding with behavior labels
        fig, ax = plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend, title=title)
        return fig, ax
    
    # convert df to list
    if not isinstance(dfs, list):
        dfs = [dfs]
    
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

#%% get traces from reference points - for interactive plotting function
def traces_from_points(behavior_dfs, behavior_variable, spread, pts):
    # concat all dfs
    behavior_dfs_all = pd.concat(behavior_dfs, keys = [num for num in range(len(behavior_dfs))])
    
    # get multi-index for selceted points: data set number and frame number
    indices = behavior_dfs_all.iloc[pts].index
    
    # loop through each selected point
    trace_data = []
    for pt_num, (df_num, frame) in enumerate(indices):
        # select appropriate data set
        df = behavior_dfs[df_num]
                
        # shift frame if near edge of data
        # if frame-spread<0:
        #     frame = spread
        #     print(f'selected frame (pt {pt_num+1}) close to edge, shifting to accommodate spread')
        # elif frame+spread>len(df)-1:
        #     frame = len(df)-1-spread
        #     print(f'selected frame (pt {pt_num+1}) close to edge, shifting to accommodate spread')
        # else:
        #     pass
        # trace = df.loc[frame-spread:frame+spread, behavior_variable]
        
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
                    
    return trace_data

#%% set axes function
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

#%% plot interactive traces
def plot_interactive_traces(n, k, UMAP_dfs, behavior_dfs, behavior_variable, spread,
                            behavior_labels = [], behavior_legend = [], sep_data=False, show_y=False):
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
        
    # convert dfs to list
    if not isinstance(UMAP_dfs, list):
        UMAP_dfs = [UMAP_dfs]
        
    if not isinstance(behavior_dfs, list):
        behavior_dfs = [behavior_dfs]
    
    # raise error if using 3D embedding
    if all('dim3' in i for i in [df.columns.tolist() for df in UMAP_dfs]):
        raise ValueError('interactive trace plotting only available for 2D UMAP embeddings')
        
    # concat all UMAP dfs
    UMAP_dfs_all = pd.concat(UMAP_dfs, keys = [num for num in range(len(UMAP_dfs))])
    # only keep dim1/dim2 for plotting
    UMAP_dfs_all = UMAP_dfs_all[['dim1','dim2']]
    
    # convert to numpy for distance calculations
    UMAP_dfs_numpy = UMAP_dfs_all.to_numpy()
    
    # raise error if number of points > total number of frames
    if n>len(UMAP_dfs_numpy):
        raise ValueError(f'n ({n}) > total points ({len(UMAP_dfs_numpy)}), pick smaller n')
    
    # plot embedding (optionally with behavior labels)      
    fig, ax = plot_embedding(UMAP_dfs, behavior_labels = behavior_labels,
                             behavior_legend = behavior_legend, sep_data=sep_data,
                             title=f'UMAP embeded points\nCHOOSE {n} POINTS')
    
    # interactively select points to show traces
    selected_pts = []
    for i in range(n):
        # user input
        pt = plt.ginput(n=1, timeout=0)
        pt = np.array(pt).squeeze()
        
        # get closest k points to selected point
        dist = np.linalg.norm(UMAP_dfs_numpy-pt, axis=1)
        sorted_dist = np.argsort(dist)
        closest_pt = UMAP_dfs_numpy[sorted_dist[:k]]
        
        # plot k selected points with text label
        ax.scatter(*closest_pt.T, c = 'k', s=10, marker = '*', label = '_'*i+'selected\npoints')
        [plt.text(*cl_pt, str(i+1)+string.ascii_letters[j]) for j, cl_pt in enumerate(closest_pt)]
        ax.set_title(f'UMAP embeded points\nselected point(s) {i+1} of {n}')

        # pause to update plot with selected point
        plt.pause(1e-6)

        selected_pts.append(closest_pt)
        
        # delete selected points so they can be selected again        
        UMAP_dfs_numpy = np.delete(UMAP_dfs_numpy, sorted_dist[:k], axis=0)        
    
    # get selected data set frame from selected points
    selected_pts = np.array(selected_pts).reshape(-1,2)
    selected_frames = [int(np.where((UMAP_dfs_all == pt).all(axis=1))[0]) for pt in selected_pts]
    
    selected_frames = np.array(selected_frames)#.reshape(n,k)

    # collect traces info from selected frames
    trace_data = traces_from_points(behavior_dfs, behavior_variable, spread, selected_frames)

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
    
    # return trace_data, selected_frames

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

trace, pts = plot_interactive_traces(2, 4, UMAP_dfs, UMAP_dfs, ['dim9','dim2'], 3, behavior_labels, behavior_legend)

print(UMAP_dfs_all.iloc[pts.flatten()])

#%%
UMAP_dfs = pd.read_csv('/Users/jimmytabet/Desktop/plotting_UMouse demo/nmf.csv')
behavior_dfs = pd.read_csv('/Users/jimmytabet/Desktop/plotting_UMouse demo/behavior.csv')
behavior_variable = ['Lpaw_Y', 'Rpaw_Y']

# out = plot_interactive_traces(5, UMAP_dfs, behavior_dfs, behavior_variable, 70)

plot_interactive_traces(3, 5, UMAP_dfs, behavior_dfs, behavior_variable, spread=70,
                            behavior_labels = [], behavior_legend = [], sep_data=True, show_y=False)

# plot_embedding(UMAP_dfs)