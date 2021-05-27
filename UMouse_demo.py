#!/usr/bin/env python
# coding: utf-8

# In[1]:

from umouse.UMouseLoader import UMouseLoader
from umouse.UMouseEstimator import UMouseEstimator
from umouse.UMousePlotter import UMousePlotter


#%%


expt_pathname = '/Users/andreagiovannucci/NEL-LAB Dropbox/NEL/Datasets/Jake/raw_data/WarrenWheelDatasets/trackingData_201218_000.mat'
output_dir = '/Users/andreagiovannucci/NEL-LAB Dropbox/NEL/Datasets/Jake/raw_data/WarrenWheelDatasets/'

# # UMouseLoader 
# Load and preprocess the data
#% initiate the class and pass it the paths for the data and output directory
um_load = UMouseLoader(expt_pathname, output_dir=output_dir)
#remove pathname definition from initilization and add to loader

# %%
#load the data, perform morlet wavelet transform, and label task events
#behavior_df, freqs, power, mwt_df, bx_labels = um_load.load_mwt_label(expt_pathname)
behavior_df = um_load.load_data(expt_pathname)

#%%
freqs, power, mwt_array = um_load.mwt(behavior_df, bodyAngle=True, jawAngle=True)
#%%
print(freqs.shape)
print(power.shape)
print(mwt_array.shape)
#%%
bx_labels = um_load.label_behavior(behavior_df)
#%%
#save the variables to a dictionary in the output_dir
um_load.save_outputs(behavior_df, freqs, power, mwt_array, bx_labels)
#%%
#initialize the estimator object
mwt_path = ['D:/data/BehaviorData/RW_data/UMouse_output/trackingData_201218_000_mwt_array.csv']
um_estimate = UMouseEstimator(mwt_path, output_dir)
#%%





# In[6]:


#Perform UMAP embedding on the processed data
um_estimate.fit() #fit_path = mwt_path


# In[7]:


um_embedding = um_estimate.transform(transform_path = mwt_path)


# # UMousePlotter
# Plot the UMAP embedding
# 

# In[ ]:


#initialize plotter class and choose datasets to include in the plot
#optionally load dlc here
#load embedding here 
embedding_paths = ['']
um_plot = UMousePlotter(embedding_paths)


# In[ ]:


#general embedding plot, can plot for each individual or single plot for group
um_plot.plot_embedding(aggregate=True)


# In[ ]:


#embedding plot labelled with behavior points, default to true, add kwargs for plotting params (alpha, etc)
#  Make more general. Pass in a vector or list or vectors if aggregate=True
um_plot.plot_bx_labels(aggregate=True, label_vector=vec)


# In[ ]:


#quiver plot, group or individual. Direction or magnitude
#will need to pass a mouse index
um_plot.quiver(aggregate=True, z_axis='direction', dlc=dlc_array)


# In[ ]:


#trajectory through umap space. option to plot trial landmarks like reward and obstacle on/off
#need args to set number of trial to plot and which trials to plot 
um_plot.umap_trajectory(landmarks=True)


# In[ ]:


#make an interactive plot that can show you limb trajectories of selected datapoints
um_plot.interactive()


# In[ ]:


um_plot.interactive_movie(mouse_ind, movie, )

