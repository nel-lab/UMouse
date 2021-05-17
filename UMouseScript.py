# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:03:04 2021

# script for using the UMouse software package

@author: Jake
"""

import UMouseLoader

pathnames = []
output_dir = []

#%% Load and preprocess the data

for expt_pathname in pathnames:
    
    #initiate the class and pass it the paths for the data and output directory
    um_load = UMouseLoader(expt_pathname, output_dir)
    
    #load the data, perform morlet wavelet transform, and label task events
    behavior_df, freqs, power, mwt_df, bx_labels = um_load.load_mwt_label(expt_pathname)
    
    #save the variables to a dictionary in the output_dir
    um_loader.save_outputs(behavior_df, freqs, power, mwt_df, bx_labels)


#%% Perform UMAP embedding on the processed data

um_estimate = UMouseEstimator(pathnames, output_dir)

um_estimate.fit(fit_path = pathnames)

um_embedding = um_estimate.transform(transform_path = pathnames)


#%% Plot the UMAP embedding



#initialize plotter class and choose datasets to include in the plot
#optionally load dlc here
#load embedding here 
um_plot = UMousePlotter(pathnames)

#general embedding plot, can plot for each individual or single plot for group
um_plot.plot_embedding(aggregate=True)

#embedding plot labelled with behavior points, default to true, add kwargs for plotting params (alpha, etc)
#  Make more general. Pass in a vector or list or vectors if aggregate=True
um_plot.bx_labels(aggregate=True, label_vector=vec)

#quiver plot, group or individual. Direction or magnitude
#will need to pass a mouse index
um_plot.quiver(aggregate=True, z_axis='direction', dlc=dlc_array)

#trajectory through umap space. option to plot trial landmarks like reward and obstacle on/off
#need args to set number of trial to plot and which trials to plot 
um_plot.umap_trajectory(landmarks=True)

#make an interactive plot that can show you limb trajectories of selected datapoints
um_plot.interactive()


um_plot.interactive_movie(mouse_ind, movie, )











