#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
#os.chdir([Umouse repo path])

from umouse.utils import WagnerLoader
from umouse import UMouse
# from umouse.UMousePlotter import UMousePlotter

#os.chdir([data path])

#%%  #Load and preprocess example data from Richard Warren's open source treadmill setup 

file_list = ['201229_000',
             '201230_000'
    ]
# '201115_000', '201217_000','201218_000', '201226_000', '201227_000',

for filename in file_list:
    expt_pathname = 'trackingData_' + filename +'.mat'
    output_dir = 'UMouseOutput/' + filename + '_behavior_df'
    
    # Load and preprocess the data
    um_load = WagnerLoader(expt_pathname, output_path=output_dir)  # 



# %%
expt_pathname = 'D:/data/BehaviorData/RW_data/201218_000/201218_000_behavior_df'


#%%
#initialize the estimator object
um_estimate = UMouse(f_sample=250, n_frequencies=25, fmin=1, fmax=None)

spect_output = um_estimate.fit_mwt()



#%%

mwt_path = ['D:/data/BehaviorData/RW_data/UMouse_output/trackingData_201218_000_mwt_array.csv']



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

