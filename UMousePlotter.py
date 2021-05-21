# -*- coding: utf-8 -*-
"""


@author: William Heffley
"""
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
            
            if isinstance(pathnames, str):
                pathnames = list([pathnames])
                
            self.bx_label_paths = bx_label_paths
            
            bx_label_list = list([])
            
            for this_path in self.pathnames:
            
                bx_labels = genfromtxt(this_path + '.csv', delimiter=',')
            
                bx_label_list.append(bx_labels)
        
    def plot_embedding(self, aggregate=True):
        #general embedding plot, can plot for each individual or single plot for group
        
    
    
    def plot_bx_labels(self, plot_list=None, labels_included=None, fig_dir=None, aggregate = False, downsample=None):
        #embedding plot labelled with behavior points, default to true, add kwargs for plotting params (alpha, etc)
        #labels included should be a list of strings or empty spaces
        
        #if no dataset is indicated for plotting then plot all the datasets
        if plot_list = None:
            plot_list = list(range(0, len(self.pathnames)))
        
        colors_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            
        #if aggregate = True then plot all points on one plot. Otherwise plot each expt separately 
        #should iterate through ax.scatter
        
        fig = plt.figure()
        ax = plt.axes(title='UMAP embeded points with behavior labels)
        
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
    
    def quiver(self, aggregate=True, z_axis='direction', dlc=dlc_array):
        #quiver plot, group or individual. Direction or magnitude
        #will need to pass a mouse index
    
    
    def umap_trajectory(self, landmarks=True):
        #trajectory through umap space. option to plot trial landmarks like reward and obstacle on/off
        #need args to set number of trial to plot and which trials to plot 
    
    
    #make an interactive plot that can show you limb trajectories of selected datapoints
    um_plot.interactive()
    
    
    
    
    