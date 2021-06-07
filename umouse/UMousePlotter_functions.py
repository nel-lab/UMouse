#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:02:23 2021

@author: jimmytabet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% plot embedding with behavior labels
def plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend, title='UMAP embeded points by behavior label'):
    # convert df to list
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    #general embedding plot, can plot for each individual or single plot for group
    fig = plt.figure()
            
    # plot 3D
    
    # if 'dim3' in dfs[0].columns:    --> shortcut, only checks if first df has dim3 
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
    
    plt.legend()
    fig.show()
    
    return fig, ax


#%% plot UMAP embedding
# def plot_embedding(self, dfs, sep_data=False):
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
    
    
    # check if behavior labels are given
    if len(behavior_labels):
        if not len(behavior_legend):
            raise ValueError('Provide behavior legend assoicated with behavior labels')
        fig, ax = plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend, title=title)
        return fig, ax
    
    # convert df to list
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    if sep_data:
        colors = None
    else:
        colors = 'C0'
    
    
    #general embedding plot, can plot for each individual or single plot for group
    fig = plt.figure()
            
    # plot 3D
    
    # if 'dim3' in dfs[0].columns:    --> shortcut, only checks if first df has dim3 
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
    
    if sep_data:
        plt.legend()
        
    fig.show()
    
    return fig, ax

#%% get traces from reference points - for interactive plotting function
def traces_from_points(behavior_dfs, behavior_variable, spread, pts):
    behavior_dfs_all = pd.concat(behavior_dfs, keys = [num for num in range(len(behavior_dfs))])
    indices = behavior_dfs_all.iloc[pts].index
    
    # loop through each selected point
    trace_data = []
    for pt_num, (df_num, frame) in enumerate(indices):
        df = behavior_dfs[df_num]
        
        # shift frame if near edge of data
        if frame-spread<0:
            frame = spread
            print(f'selected frame (pt {pt_num+1}) close to edge, shifting to accommodate spread')
        elif frame+spread>len(df)-1:
            frame = len(df)-1-spread
            print(f'selected frame (pt {pt_num+1}) close to edge, shifting to accommodate spread')
        else:
            pass

        # store data in traces variable
        pt_dict = {'df_num': df_num+1,
                   'frame': frame,
                   'trace': df.loc[frame-spread:frame+spread, behavior_variable]}
        trace_data.append(pt_dict)
                    
    return trace_data

#%% plot interactive traces
def plot_interactive_traces(n, UMAP_dfs, behavior_dfs, behavior_variable, spread,
                            behavior_labels = [], behavior_legend = [], sep_data=False):
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
        
    UMAP_dfs_all = pd.concat(UMAP_dfs, keys = [num for num in range(len(UMAP_dfs))])
    UMAP_dfs_all = UMAP_dfs_all[['dim1','dim2']]
    
    UMAP_dfs_numpy = UMAP_dfs_all.to_numpy()
    
    if n>len(UMAP_dfs_numpy):
        raise ValueError(f'n ({n}) > total points ({len(UMAP_dfs_numpy)}), pick smaller n')
    
    # plot embedding        
    fig, ax = plot_embedding(UMAP_dfs, behavior_labels = behavior_labels, behavior_legend = behavior_legend, sep_data=sep_data, title=f'UMAP embeded points\nCHOOSE {n} POINTS')
    
    # get user input for n points
    pts = plt.ginput(n)
    pts = np.array(pts).squeeze()
            
    # select closest point to user input and plot
    selected_pts = []
    for i, pt in enumerate(pts):
        dist = np.linalg.norm(UMAP_dfs_numpy-pt, axis=1)
        sorted_dist = np.argsort(dist)
        closest_pt = UMAP_dfs_numpy[sorted_dist[0]]
        selected_pts.append(closest_pt)
        
        UMAP_dfs_numpy = np.delete(UMAP_dfs_numpy, sorted_dist[0], axis=0)
        
        ax.scatter(*closest_pt, c = 'k', marker = '*', label = '_'*i+'selected\npoints')
        plt.text(*closest_pt, str(i+1))
    
    ax.set_title(f'UMAP embeded points\n{n} selected points')
    plt.legend()
    fig.show()
    
    selected_pts = np.array(selected_pts)

    selected_frames = [int(np.where((UMAP_dfs_all == pt).all(axis=1))[0]) for pt in selected_pts]
    
    trace_data = traces_from_points(behavior_dfs, behavior_variable, spread, selected_frames)
  
    fig2 = plt.figure()
    fig2.set_tight_layout(True)
    fig2.suptitle('Traces of Selected Points')

    for num, pt in enumerate(trace_data):
        ax = fig2.add_subplot(n, 1, num+1)
        ax.plot(pt['trace'])
        if sep_data:
            ax.set_title(f'Point {num+1}, df_{pt["df_num"]}')
        else:
            ax.set_title(f'Point {num+1}')
        ax.set_xticks([pt['frame']])
    
        if num == n-1:
            ax.set_xlabel('FRAME', size='large')
            
    fig2.legend(behavior_variable)

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
print(UMAP_dfs_all)

# plot_embedding(UMAP_dfs, behavior_labels, behavior_legend)

plot_interactive_traces(5, UMAP_dfs, UMAP_dfs, ['dim9','dim2'], 3, behavior_labels, behavior_legend)

#%%
UMAP_dfs = pd.read_csv('/Users/jimmytabet/Desktop/plotting_UMouse demo/nmf.csv')
behavior_dfs = pd.read_csv('/Users/jimmytabet/Desktop/plotting_UMouse demo/behavior.csv')
behavior_variable = ['Lpaw_Y', 'Rpaw_Y']

out = plot_interactive_traces(5, UMAP_dfs, behavior_dfs, behavior_variable, 70)

# plot_embedding(UMAP_dfs)
