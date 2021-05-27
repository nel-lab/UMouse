# -*- coding: utf-8 -*-
"""


@author: William Heffley
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class UMousePlotter:
    def __init__(self, pathnames, bx_label_paths=None):
        #initialize plotting class and choose datasets to include in the plot
        #pathnames for the umap embedded points
        
        if isinstance(pathnames, str):
            pathnames = list([pathnames])
            
        # save pathnames as a class field
        self.pathname = pathnames
        
        #get the filename for experiemnt
        self.filename = os.path.basename(pathnames) 
        
        #get directory for each experiment
        self.dirname = os.path.dir(pathnames)
        
        embedding_list = list([])
        
        #load embeddings here and group into a list
        for this_path in self.pathnames:
            
            umap_embedding = genfromtxt(this_path + '.csv', delimiter=',')
            
            embedding_list.append(umap_embedding)
        
        self.embedding_list = embedding_list    
        
        #optionally load the behavior labels generated in UMouseLoader
        if bx_label_paths:
            
            if isinstance(bx_label_paths, str):
                bx_label_paths = list([bx_label_paths])
                
            self.bx_label_paths = bx_label_paths
            
            bx_label_list = list([])
            
            for this_path in self.bx_label_paths:
            
                bx_labels = genfromtxt(this_path + '.csv', delimiter=',')
            
                bx_label_list.append(bx_labels)
                
            self.bx_labels_list = bx_labels_list
        
    def plot_embedding(self, dfs, sep_data=False):
        '''

        Parameters
        ----------
        dfs : dataframe or list of dataframes
            Data to be plotted.
        sep_data : bool, optional
            Display each dataset in different color. The default is False.

        Returns
        -------
        None.

        '''
        
        a = np.random.rand(5,3)
        b = np.random.rand(5,3)
        c = np.random.rand(5,3)
        a_df = pd.DataFrame(a, columns = ['dim1','dim2','other'])
        b_df = pd.DataFrame(b, columns = ['dim1','dim2','other'])
        c_df = pd.DataFrame(c, columns = ['dim1','dim2','other'])
        
        
        
        dfs = [a_df, b_df, c_df]
        sep_data=True
        
        
        
        #######
        # convert df to list
        if not isinstance(dfs, list):
            dfs = list(dfs)
        
        if sep_data:
            colors = None
        else:
            colors = 'C0'
        
        
        #general embedding plot, can plot for each individual or single plot for group
        fig = plt.figure()
                
        # plot 3D
        if 'dim3' in dfs[0].columns:        
            ax = fig.add_subplot(projection='3d', title='UMAP embeded points',
                                 xlabel='dimension 1', ylabel = 'dimension 2', zlabel = 'dimension 3')
            for num, df in enumerate(dfs):
                ax.scatter(df['dim1'], df['dim2'], df['dim3'], c = colors)
        # plot 2D
        else:
            ax = fig.add_subplot(title='UMAP embeded points', xlabel='dimension 1', ylabel = 'dimension 2')
            for num, df in enumerate(dfs):
                ax.scatter(df['dim1'], df['dim2'], c = colors)

        
                
                
                
                
                
        fig.show()
    
    def plot_bx_labels(self, plot_list=None, labels_included=None, fig_dir=None, 
                       aggregate = False, downsample=None, kwargs_dict=None):
        #embedding plot labelled with behavior points, default to true, add kwargs for plotting params (alpha, etc)
        #plot_list needs to be a set of indeces for which experiments to plot
        #labels included should be a list of strings or empty spaces
        #kwargs_dict is a dictionary of kwargs for the scatterplot options (c, marker, etc..)
        
        #if no dataset is indicated for plotting then plot all the datasets
        if plot_list == None:
            plot_list = list(range(0, len(self.pathnames)))
        
        colors_list = ['k', 'g', 'r', 'c', 'm', 'b', 'y']
        
        #check for plotting kwargs or set default values
        if kwargs_dict == None:
                kwargs_dict = {'marker':'o', 's':'0.2', 'alpha':'0.2'}
        
        #if agg is false then create single figure/plot for all experiments in plot_list
        if aggregate==True:
            fig = plt.figure()
            ax = plt.axes(title='UMAP embeded points with behavior labels')
            
        for expt_ind in plot_list:
            
            this_bx_vec = self.bx_labels_list[expt_ind]
            this_embedding = self.embedding_list[expt_ind]
            
            #trim because embedding has n-1 frames compared to bx_labels due to Morlet wavelet transform
            if len(this_bx_vec) == len(this_embedding) + 1:
                this_bx_vec = this_bx_vec[0:-1] 
                
            #generate figure and title for each experiment only if agg is false
            if aggregate==False:
                fig = plt.figure()
                ax = plt.axes(title='UMAP embeded points with behavior labels: ' + self.filename[expt_ind])
                
            #plot data points
            for label_ind, label_str in enumerate(labels_included):
                    
                    #plot the embedded points
                    ax.scatter(*self.this_embedding.T[:,[np.where(this_bx_vec==label_ind)[0]]], 
                               label=label_str, 
                               c=colors_list[label_ind], 
                               **kwargs_dict)
            
            if aggregate == False:
                #set axes and legend
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.legend()
                
                #save figure
                if fig_dir:
                    fig.savefig(fig_dir + self.filename)
                    
        if aggregate == True:
                #set axes and legend
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.legend()
                
                #save figure
                if fig_dir:
                    fig.savefig(fig_dir + 'aggregate_plot')
    
    def quiver(self, aggregate=True, z_axis='direction', dlc=dlc_array):
        #quiver plot, group or individual. Direction or magnitude
        #will need to pass a mouse index
        fig = plt.figure()
    
    def umap_trajectory(self, landmarks=True):
        #trajectory through umap space. option to plot trial landmarks like reward and obstacle on/off
        #need args to set number of trial to plot and which trials to plot 
        fig = plt.figure()
    
    #make an interactive plot that can show you limb trajectories of selected datapoints
    um_plot.interactive()
    
    
    
    
    