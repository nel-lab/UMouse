#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:02:23 2021

@author: jimmytabet

Plotting module for UMouse.
"""

#%% imports
import os, cv2, h5py, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.util import montage

#%% functions
def load_dfs(dfs):
    '''
    Load dataframes.

    Parameters
    ----------
    dfs : str/pandas df, or list of str's/pandas df's
        Dataframe(s) to be loaded.

    Returns
    -------
    dfs_all : list
        List of loaded pandas dataframe(s).

    '''
    
    # convert to list
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    # iterate through each input and append to list
    dfs_all = []
    for df in dfs:
        if isinstance(df, str):
            dfs_all.append(pd.read_csv(df))
        else:
            dfs_all.append(pd.DataFrame(df))
    
    return dfs_all

def plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend, ds=1, save=False):
    '''
    Plot UMAP embedding with behavior labels.

    Parameters
    ----------
    dfs : str/pandas df, or list of str's/pandas df's
        UMAP embeddings to be plotted.
    behavior_labels : str/pandas df/numpy array, or list of str's/pandas df's/numpy arrays
        Behavior labels associated with each UMAP points. Each individual label
        should be an integer from 0 to (number of behavior types).
    behavior_legend : list
        Legend associated with behavior_labels, each entry describing the
        behavior_label at that index (i.e. ['walk','jump','stop'] assigns 'walk'
        to behavior_labels equal to 0, 'jump' to 1, and 'stop' to 2).
    ds : int, optional
        Plot every 'ds' points (i.e. downsample). The default is 1, which plots
        every point.
    save : bool or str, optional
        Save UMAP embedding. If a str is passed, the figure will be saved with 
        that name/location. The default is False. 

    Returns
    -------
    fig : matplotlib figure
        UMAP embedding figure.
    ax : matplotlib axes
        UMAP embedding axes.

    '''
    
    # load dfs
    dfs = load_dfs(dfs)
    behavior_labels = load_dfs(behavior_labels)

    # check that same number of data sets are passed to UMAP and behavior_labels
    if len(dfs) != len(behavior_labels):
        raise ValueError('Number of data sets in UMAP and behavior labels does not match')

    # check that UMAP and behavior_labels are same size
    elif not all([len(i) == len(j) for i,j in zip(dfs, behavior_labels)]):
        raise ValueError('Size of UMAP points and corresponding behavior labels does not match')
    else:
        pass
    
    # downsample
    dfs = [df[::ds] for df in dfs]
    behavior_labels = [df[::ds] for df in behavior_labels]
    behavior_labels = pd.concat(behavior_labels).to_numpy()
    
    fig = plt.figure()
            
    # plot 3D
    if all('dim_2' in i for i in [df.columns.tolist() for df in dfs]):   
        dfs = pd.concat(dfs)

        ax = fig.add_subplot(projection='3d', title='UMAP embeded points by behavior label',
                             xlabel='dimension 1', ylabel = 'dimension 2', zlabel = 'dimension 3')
        
        # label by behavior
        for lab in np.unique(behavior_labels):
            df = dfs[behavior_labels == lab]

            ax.scatter(df['dim_0'], df['dim_1'], df['dim_2'], label = behavior_legend[lab])
   
    # plot 2D
    else:
        dfs = pd.concat(dfs)

        ax = fig.add_subplot(title='UMAP embeded points by behavior label', xlabel='dimension 1', ylabel = 'dimension 2')
        
        # label by behavior
        for lab in np.unique(behavior_labels):
            df = dfs[behavior_labels == lab]

            ax.scatter(df['dim_0'], df['dim_1'], label = behavior_legend[lab])
    
    fig.legend()
    
    # save
    if isinstance(save, str):
        plt.savefig(save, dpi=300)
    elif save:
        plt.savefig('UMAP_embedding_behavior_labels', dpi=300)
    else:
        pass
    
    return fig, ax

def plot_embedding(dfs, behavior_labels = [], behavior_legend = [], ds=1, sep_data=False, save=False):
    '''
    Plot UMAP embedding.

    Parameters
    ----------
    dfs : str/pandas df, or list of str's/pandas df's
        UMAP embeddings to be plotted.    
    behavior_labels : str/pandas df/numpy array, or list of str's/pandas df's/numpy arrays, optional
        Behavior labels associated with each UMAP points. Each individual label
        should be an integer from 0 to (number of behavior types). If provided,
        will plot UMAP embedding by labels. The default is [], which only plots
        embedding.
    behavior_legend : list, optional
        Legend associated with behavior_labels, each entry describing the
        behavior_label at that index (i.e. ['walk','jump','stop'] assigns 'walk'
        to behavior_labels equal to 0, 'jump' to 1, and 'stop' to 2). If not provided
        when passing in behavior_labels, legend will automaically be generated
        in the form ['behavior 1', behavior 2',...]. The default is [], legend 
        automatically generated if passing in behavior_labels.
    ds : int, optional
        Plot every 'ds' points (i.e. downsample). The default is 1, which plots
        every point.
    sep_data : bool, optional
        Display each dataset in different color. The default is False.
    save : bool or str, optional
        Save UMAP embedding. If a str is passed, the figure will be saved with 
        that name/location. The default is False. 

    Returns
    -------
    fig : matplotlib figure
        UMAP embedding figure.
    ax : matplotlib axes
        UMAP embedding axes.

    '''
    
    # plot embedding with behavior labels if given
    if len(behavior_labels):
        # generate behavior_legend if none provided
        if not len(behavior_legend):
            behavior_legend = [f'behavior {num+1}' for num in range(max(np.unique(behavior_labels))+1)]
        
        # plot embedding with behavior labels
        fig, ax = plot_embedding_behavior_labels(dfs, behavior_labels, behavior_legend,
                                                 ds=ds, save=save)
        return fig, ax
    
    # load dfs and downsample
    dfs = load_dfs(dfs)
    dfs = [df[::ds] for df in dfs]
    
    # change color per data set of leave as one color
    if sep_data:
        colors = None
    else:
        colors = 'C0'
    
    fig = plt.figure()
            
    # plot 3D
    if all('dim_2' in i for i in [df.columns.tolist() for df in dfs]):    
        ax = fig.add_subplot(projection='3d', title='UMAP embeded points',
                             xlabel='dimension 1', ylabel = 'dimension 2', zlabel = 'dimension 3')
        for num, df in enumerate(dfs):
            # seperate data labels
            if sep_data:
                label = f'df_{num+1}'
            # same data labels
            else:
                label = '_'*num+'data'
            
            ax.scatter(df['dim_0'], df['dim_1'], df['dim_2'], c = colors, label = label)
   
    # plot 2D
    else:
        ax = fig.add_subplot(title='UMAP embeded points', xlabel='dimension 1', ylabel = 'dimension 2')
        for num, df in enumerate(dfs):
            # seperate data labels
            if sep_data:
                label = f'df_{num+1}'
            # same data labels
            else:
                label = '_'*num+'data'
            
            ax.scatter(df['dim_0'], df['dim_1'], c = colors, label = label)
    
    # show legend if seperate data sets
    if sep_data:
        fig.legend()
    
    # save
    if isinstance(save, str):
        plt.savefig(save, dpi=300)
    elif save:
        plt.savefig('UMAP_embedding', dpi=300)
    else:
        pass
    
    return fig, ax

def set_axes(figure, num_clusters, num_points, show_y = False):
    '''
    Set equal y-axes for each cluster when plotting traces.

    Parameters
    ----------
    figure : matplotlib figure
        Traces figure.
    num_clusters : int
        Number of clusters.
    num_points : int
        Number of points in each cluster.
    show_y : bool, optional
        Show y-axes values. The default is False.

    Returns
    -------
    None.

    '''
    
    # get subplots in figure
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
        
def get_points_high_dim(manual_points, k, spread, UMAP_dfs, ds):
    '''
    Generate k_neighbors to each point in manual_points. Used in high dimension
    UMAP embedding (>3) so as to not plot points as in get_points function.

    Parameters
    ----------
    manual_points : list or numpy array
        List or array of points, from which k_neighbors will be generated. Points
        should have same dimension as columns in UMAP_embedding.
    k : int
        Number of points in each cluster.
    spread : int
        How many frames (before and after) each point to plot, i.e.
        [point-spread:point+spread].
    UMAP_dfs : str/pandas df, or list of str's/pandas df's
        UMAP embeddings to be plotted.

    Returns
    -------
    selected_frames : numpy array
        Array of frame indices at the user-selected points, corresponding to 
        index of UMAP/behavior data (concatenated if multiple datasets are passed).

    '''
    
    n = len(manual_points)
              
    # downsample UMAP
    UMAP_dfs_ds = [df[::ds] for df in UMAP_dfs]
        
    # concat all UMAP dfs
    UMAP_dfs_all_og = pd.concat(UMAP_dfs_ds, keys = [num for num in range(len(UMAP_dfs_ds))])
    # only keep dim_1/dim_2 for plotting
    # UMAP_dfs_all_og = UMAP_dfs_all_og[['dim_1','dim_2']]
    
    UMAP_dfs_all = UMAP_dfs_all_og.copy()
    
    # raise error if number of points > total number of frames
    if (n*k)>len(UMAP_dfs_all):
        if ds != 1:
            raise ValueError(f'n_clusters*k_neighbors ({n*k}) > total downsampeled points ({len(UMAP_dfs_all)}). Decrease n_clusters, k_neighbors, or ds')
        else:
            raise ValueError(f'n_clusters*k_neighbors ({n*k}) > total points ({len(UMAP_dfs_all)}). Decrease n_clusters or k_neighbors.')

    # select points to show traces
    selected_pts_all = []
    for i in range(n):
        pt = manual_points[i]
        
        # sort points by distance to selected point
        dist = np.linalg.norm(UMAP_dfs_all-pt, axis=1)
        sorted_dist = np.argsort(dist)
        # get sorted point indices
        index = UMAP_dfs_all.iloc[sorted_dist].index
        
        # init variables to pick points that are not close or on edge of data
        selected_pts = []
        idxs = []

        # iter through all points to find suitable ones
        for idx in index:
            # check if there are k points yet
            if len(selected_pts) == k:
                break
            
            # init variables
            (df_num, frame) = idx
            skip = False
            end = len(UMAP_dfs[df_num])
            
            # check if close to previously selected points
            if len(idxs):
                sel_pt = np.array(idxs)
                sel_pt_df = sel_pt[sel_pt[:,0] == df_num]
                
                check = abs(sel_pt_df[:,1]-frame)<spread
                skip = check.any()
                
            # edge cases - skip point
            if frame-spread<0 or frame+spread>end-1:
                continue
            # point close to previously selected point - skip point
            elif skip:
                continue
            # add to list and remove to not duplicate
            else:
                idxs.append(idx)
                selected_pts.append(UMAP_dfs_all.loc[idx])
                UMAP_dfs_all = UMAP_dfs_all.drop(idx)
        
        # if not enough points, do it again without restrictions on proximity to previous points
        if len(selected_pts) != k:
    
            # iter through all points to find suitable ones
            for idx in index:
                if idx in idxs:
                    continue

                # check if there are k points yet
                if len(selected_pts) == k:
                    break
                
                # init variables
                (df_num, frame) = idx
                end = len(UMAP_dfs[df_num])
                    
                # edge cases - skip point
                if frame-spread<0 or frame+spread>end-1:
                    continue
                # add to list and remove to not duplicate
                else:
                    selected_pts.append(UMAP_dfs_all.loc[idx])
                    UMAP_dfs_all = UMAP_dfs_all.drop(idx)
        
        # check if there are enough usable points
        selected_pts = np.array(selected_pts)
        if len(selected_pts) != k:
            raise ValueError('Not enough eligible points. Decrease n_clusters, k_neighbors, or spread')
                        
        # append to all points list            
        selected_pts_all.append(selected_pts)
    
    # get selected frames/indices from selected points
    selected_pts_all = np.array(selected_pts_all).reshape(-1, UMAP_dfs_all_og.shape[1])
    selected_frames = [int(np.where((UMAP_dfs_all_og == pt).all(axis=1))[0]) for pt in selected_pts_all]
    
    # multiply by ds to get proper frames/indices
    selected_frames = ds*np.array(selected_frames)
    
    return selected_frames

#%% interactive plotting class
class interactive():
    """
    Class for interactive plotting.

    Attributes
    ----------
    n_clusters : int
        Number of clusters.
    k_neighbors : int
        Number of points in each cluster.
    spread : int
        How many frames (before and after) each point to plot, i.e.
        [point-spread:point+spread].
    UMAP_dfs : str/pandas df, or list of str's/pandas df's
        UMAP embeddings to be plotted.
    behavior_labels : str/pandas df/numpy array, or list of strs/pandas dfs/numpy arrays
        Behavior labels associated with each UMAP points. Each individual label
        should be an integer from 0 to (number of behavior types). If provided,
        will plot UMAP embedding by labels.
    behavior_legend : list
        Legend associated with behavior_labels, each entry describing the
        behavior_label at that index (i.e. ['walk','jump','stop'] assigns 'walk'
        to behavior_labels equal to 0, 'jump' to 1, and 'stop' to 2). If not provided
        when passing in behavior_labels, legend will automaically be generated
        in the form ['behavior 1', behavior 2',...].
    ds : int
        Plot every 'ds' points (i.e. downsample).
    selected_frames : numpy array
        Array of frame indices at the user-selected points, corresponding to 
        index of UMAP/behavior data (concatenated if multiple datasets are passed).

    Methods
    -------
    get_points(sep_data=False, save_embedding=False, save_chosen_points=False):
        Choose points interactively from UMAP embedding, or generate k_neighbors if passed a list of points.
    plot_traces(behavior_dfs, behavior_variable, selected_frames = [], show_y=False, save=False):
        Plot traces for selected frames.
    behavior_montage(video_path, save_path, fps, indices = []):  
        Create and save behavior montage for selected frames.

    """
    
    def __init__(self, n_clusters, k_neighbors, spread, UMAP_dfs, behavior_labels = [], behavior_legend = [], ds = 1):        
        '''
        Initialization.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        k_neighbors : int
            Number of points in each cluster.
        spread : int
            How many frames (before and after) each point to plot, i.e.
            [point-spread:point+spread].
        UMAP_dfs : str/pandas df, or list of str's/pandas df's
            UMAP embeddings to be plotted.
        behavior_labels : str/pandas df/numpy array, or list of str's/pandas df's/numpy arrays, optional
            Behavior labels associated with each UMAP points. Each individual label
            should be an integer from 0 to (number of behavior types). If provided,
            will plot UMAP embedding by labels. The default is [], which only plots
            embedding.
        behavior_legend : list, optional
            Legend associated with behavior_labels, each entry describing the
            behavior_label at that index (i.e. ['walk','jump','stop'] assigns 'walk'
            to behavior_labels equal to 0, 'jump' to 1, and 'stop' to 2). If not provided
            when passing in behavior_labels, legend will automaically be generated
            in the form ['behavior 1', behavior 2',...]. The default is [], legend 
            automatically generated if passing in behavior_labels.
        ds : int, optional
            Plot every 'ds' points (i.e. downsample). The default is 1, which plots
            every point.

        Returns
        -------
        None.

        '''
        
        # load dfs    
        UMAP_dfs = load_dfs(UMAP_dfs)
        behavior_labels = load_dfs(behavior_labels)
        
        # if behavior labels are passed in, check that they match size of UMAP
        if len(behavior_labels):
            # check that same number of data sets are passed to UMAP and behavior_labels
            if len(UMAP_dfs) != len(behavior_labels):
                raise ValueError('Number of data sets in UMAP and behavior labels does not match')
        
            # check that UMAP and behavior_labels are same size
            elif not all([len(i) == len(j) for i,j in zip(UMAP_dfs, behavior_labels)]):
                raise ValueError('Size of UMAP points and corresponding behavior labels does not match')
            else:
                pass
        
        # set class attributes
        self.n = n_clusters
        self.k = k_neighbors
        self.spread = spread
        self.UMAP_dfs = UMAP_dfs
        self.behavior_labels = behavior_labels
        self.behavior_legend = behavior_legend
        self.ds = ds
        
    def get_points(self, manual_points = [], sep_data=False, save_embedding=False, save_chosen_points=False):
        '''
        Choose points interactively from UMAP embedding, or generate k_neighbors if passed a list of points.

        Parameters
        ----------
        manual_points : list or numpy array, optional
            List or array of points, from which k_neighbors will be generated.
            Points should have same dimension as columns in UMAP_embedding.
            The default is [], which prompts user to interactively choose points
            from UMAP embedding (if 2D).
        sep_data : bool, optional
            Display each dataset in different color. The default is False.
        save_embedding : bool or str, optional
            Save UMAP embedding. If a str is passed, the figure will be saved with 
            that name/location. The default is False. 
        save_chosen_points : bool or str, optional
            Save UMAP embedding with chosen points shown. If a str is passed,
            the figure will be saved with that name/location. The default is False. 

        Returns
        -------
        selected_frames : numpy array
            Array of frame indices at the user-selected points, corresponding to 
            index of UMAP/behavior data (concatenated if multiple datasets are passed).

        '''
        
        # set function variable to class attributes
        n = self.n
        k = self.k
        spread = self.spread
        UMAP_dfs = self.UMAP_dfs
        behavior_labels = self.behavior_labels
        behavior_legend = self.behavior_legend
        ds = self.ds
        
        # check for compatibility 
        if len(manual_points):
            dim_check = np.array(manual_points)
            
            # raise error if dimension does not equal UMAP dimension
            if dim_check.shape[1] != UMAP_dfs[0].shape[1]:
                raise ValueError(f'dimension of manual_points ({dim_check.shape[1]}) not equal to dimension of UMAP_embedding ({UMAP_dfs[0].shape[1]})')
            
            # use get_points_high_dim if dim>3 (so as to not plot)
            elif dim_check.shape[1] > 3:
                return get_points_high_dim(manual_points, k, spread, UMAP_dfs, ds)
            
            # pass if everything is fine
            else:
                pass
        
        # raise error if using 3D embedding and did not provide points
        else:
            if all('dim_2' in i for i in [df.columns.tolist() for df in UMAP_dfs]):
                raise ValueError('interactive trace plotting only available for 2D UMAP embeddings')
                  
        # downsample UMAP
        UMAP_dfs_ds = [df[::ds] for df in UMAP_dfs]
            
        # concat all UMAP dfs
        UMAP_dfs_all_og = pd.concat(UMAP_dfs_ds, keys = [num for num in range(len(UMAP_dfs_ds))])
        # only keep dim_1/dim_2 for plotting
        # UMAP_dfs_all_og = UMAP_dfs_all_og[['dim_1','dim_2']]
        
        UMAP_dfs_all = UMAP_dfs_all_og.copy()
        
        # raise error if number of points > total number of frames
        if (n*k)>len(UMAP_dfs_all):
            if ds != 1:
                raise ValueError(f'n_clusters*k_neighbors ({n*k}) > total downsampeled points ({len(UMAP_dfs_all)}). Decrease n_clusters, k_neighbors, or ds')
            else:
                raise ValueError(f'n_clusters*k_neighbors ({n*k}) > total points ({len(UMAP_dfs_all)}). Decrease n_clusters or k_neighbors.')
        
        # plot embedding (optionally with behavior labels)      
        fig, ax = plot_embedding(UMAP_dfs, behavior_labels = behavior_labels,
                                 behavior_legend = behavior_legend, ds=ds,
                                 sep_data=sep_data, save = save_embedding)

        # set title to instruct user to choose points        
        ax.set_title(f'UMAP embeded points\nCHOOSE {n} POINTS')
        
        # interactively select points to show traces
        selected_pts_all = []
        for i in range(n):
            if not len(manual_points):
                # user input
                pt = plt.ginput()
                pt = np.array(pt).squeeze()
            else:
                pt = manual_points[i]
            
            # sort points by distance to selected point
            dist = np.linalg.norm(UMAP_dfs_all-pt, axis=1)
            sorted_dist = np.argsort(dist)
            # get sorted point indices
            index = UMAP_dfs_all.iloc[sorted_dist].index
            
            # init variables to pick points that are not close or on edge of data
            selected_pts = []
            idxs = []
    
            # iter through all points to find suitable ones
            for idx in index:
                # check if there are k points yet
                if len(selected_pts) == k:
                    break
                
                # init variables
                (df_num, frame) = idx
                skip = False
                end = len(UMAP_dfs[df_num])
                
                # check if close to previously selected points
                if len(idxs):
                    sel_pt = np.array(idxs)
                    sel_pt_df = sel_pt[sel_pt[:,0] == df_num]
                    
                    check = abs(sel_pt_df[:,1]-frame)<spread
                    skip = check.any()
                    
                # edge cases - skip point
                if frame-spread<0 or frame+spread>end-1:
                    continue
                # point close to previously selected point - skip point
                elif skip:
                    continue
                # add to list and remove to not duplicate
                else:
                    idxs.append(idx)
                    selected_pts.append(UMAP_dfs_all.loc[idx])
                    UMAP_dfs_all = UMAP_dfs_all.drop(idx)
            
            # if not enough points, do it again without restrictions on proximity to previous points
            if len(selected_pts) != k:
        
                # iter through all points to find suitable ones
                for idx in index:
                    if idx in idxs:
                        continue

                    # check if there are k points yet
                    if len(selected_pts) == k:
                        break
                    
                    # init variables
                    (df_num, frame) = idx
                    end = len(UMAP_dfs[df_num])
                        
                    # edge cases - skip point
                    if frame-spread<0 or frame+spread>end-1:
                        continue
                    # add to list and remove to not duplicate
                    else:
                        selected_pts.append(UMAP_dfs_all.loc[idx])
                        UMAP_dfs_all = UMAP_dfs_all.drop(idx)
            
            # check if there are enough usable points
            selected_pts = np.array(selected_pts)
            if len(selected_pts) != k:
                raise ValueError('Not enough eligible points. Decrease n_clusters, k_neighbors, or spread')
                            
            # plot k selected points with text label
            ax.scatter(*selected_pts.T, c = 'k', s=10, marker = '*', label = '_'*i+'selected\npoints')
            [ax.text(*sl_pt, str(i+1)+string.ascii_letters[j]) for j, sl_pt in enumerate(selected_pts)]
            ax.set_title(f'UMAP embeded points\nselected point(s) {i+1} of {n}')
    
            # pause to update plot with selected point
            plt.pause(1e-6)
            
            # append to all points list            
            selected_pts_all.append(selected_pts)
        
        # get selected frames/indices from selected points
        selected_pts_all = np.array(selected_pts_all).reshape(-1, UMAP_dfs_all_og.shape[1])
        selected_frames = [int(np.where((UMAP_dfs_all_og == pt).all(axis=1))[0]) for pt in selected_pts_all]
        
        # multiply by ds to get proper frames/indices
        selected_frames = ds*np.array(selected_frames)
        
        # store as class attribute
        self.selected_frames = selected_frames
        
        # save
        if isinstance(save_chosen_points, str):
            plt.savefig(save_chosen_points, dpi=300)
        elif save_chosen_points:
            plt.savefig('UMAP_embedding_with_points', dpi=300)
        else:
            pass
        
        return selected_frames
    
    def plot_traces(self, behavior_dfs, behavior_variable, selected_frames = [], show_y=False, save=False):
        '''
        Plot traces for selected frames.

        Parameters
        ----------
        behavior_dfs : str/pandas df, or list of str's/pandas df's
            Behavior data sets to be plotted.
        behavior_variable : str or list
            Body part or list of parts to be plotted. Each entry should be column
            in behavior_dfs.
        selected_frames : numpy array, optional
            Array of frame indices, corresponding to index of UMAP/behavior data
            (concatenated if multiple datasets are passed). The default is [], 
            which will prompt the user to interactively selected points.
        show_y : bool, optional
            Show y-axes values. The default is False.
        save : bool or str, optional
            Save traces. If a str is passed, the figure will be saved with 
            that name/location. The default is False. 

        Returns
        -------
        None.

        '''
    
        # load dfs    
        behavior_dfs = load_dfs(behavior_dfs)
        
        # check that same number of data sets are passed to UMAP and behavior dfs
        if len(self.UMAP_dfs) != len(behavior_dfs):
            raise ValueError('Number of data sets in UMAP and behavior data does not match')    
    
        # get interactive points from UMAP embedding
        if not len(selected_frames):
            if not hasattr(self, 'selected_frames'):
                self.get_points()

            selected_frames = self.selected_frames
        
        # set function variable to class attributes
        n = self.n
        k = self.k
        spread = self.spread

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
            
            # get trace
            trace = df.loc[frame-spread:frame+spread, behavior_variable]
    
            # add empty points if frame is near edge of data - should not happen
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
            # add point number (and data set number if multiple)
            if len(behavior_dfs) > 1:
                ax.set_title(f'Point {num//k+1}{string.ascii_letters[num%k]}, df_{pt["df_num"]}')
            else:
                ax.set_title(f'Point {num//k+1}{string.ascii_letters[num%k]}')
            
            # show beginning, middle, and end frames            
            ax.set_xticks([pt['trace'].index[pt['trace'].index>=0][0],
                           pt['frame'],
                           pt['trace'].index[pt['trace'].notnull().all(1)].max()])
            
            # alternatively, only show middle frame
            # ax.set_xticks([pt['frame']])
            
            # show entire spread even if near edge
            ax.set_xlim(pt['frame']-spread, pt['frame']+spread)
            
        # make ylim the same for each cluster of points
        set_axes(fig2, n, k, show_y=show_y)
        
        fig2.legend(behavior_variable)
        
        # save
        if isinstance(save, str):
            plt.savefig(save, dpi=300)
        elif save:
            plt.savefig('traces_of_chosen_points', dpi=300)
        else:
            pass
        
    def behavior_montage(self, video_path, save_path, fps, indices = []):        
        '''
        Create and save behavior montage for selected frames.
    
        Parameters
        ----------
        video_path : str
            Path to behavior video. Must be hdf5.
        save_path : str
            Path to save behavior montage. Must include extension/format (.avi, .mp4, .mov).
        fps : int
            Frame rate of the behavior video.
        indices : numpy array
            Array of frame indices, corresponding to index of UMAP/behavior data
            (concatenated if multiple datasets are passed). The default is [], 
            which will prompt the user to interactively selected points.
    
        Returns
        -------
        None. Saves behavior montage to save_path.
    
        '''
        
        # raise error if more than 1 dataset
        if len(self.UMAP_dfs) != 1:
            raise ValueError('Behavior montage only works with 1 dataset/video')
            
        # raise error if save_path does not include extension
        if len(os.path.splitext(save_path)[-1]) == 0:
            raise ValueError('Specify format/extension for save_path')
        
        # get interactive points from UMAP embedding
        if not len(indices):
            if not hasattr(self, 'selected_frames'):
                self.get_points()

            indices = self.selected_frames
                
        # set function variable to class attributes
        n = self.n
        k = self.k
        spread = self.spread
                
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
        
        # # check if file opened successfully
        # if (cap.isOpened()== False): 
        #     raise ValueError('Error opening video file')
        
        # # read in video
        # while(cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret == True:
        #         # show resulting frame
        #         cv2.imshow('Frame',frame)
        #         # press 'q' to exit
        #         if cv2.waitKey(25) & 0xFF == ord('q'):
        #             break          
        #     else: 
        #       break
        
        # # close
        # cap.release()
        # cv2.destroyAllWindows()