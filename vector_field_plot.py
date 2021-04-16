# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:13:12 2021

@author: Jake
"""
#%% dependencies and paths 

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from numpy import genfromtxt
from scipy import interpolate

#make a list of datasets
data_fn_list  = list(['201115_000', 
                     '201217_000', '201218_000', '201226_000', '201227_000', 
                     '201228_000', '201229_000', '201230_000'])

#select the dataset to analyze
# data_fn = data_fn_list[2]
# print(data_fn)

#set path
# data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'
embed_date = '210331'

#%% vector field plotting 

xx = np.array([])
yy = np.array([])
uu = np.array([])
vv = np.array([])

for data_fn in data_fn_list:
    data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'
    
    down_samp = 100
    cmap = mpl.cm.hsv #cyclical cmap for angle
    
    #load data points 
    umap_embedding = genfromtxt(data_dir + data_fn + '_multiMouseEmbed_50k_' + embed_date + '.csv', 
                                delimiter=',')
    
    #downsample and calculate vector components
    x = umap_embedding[0::down_samp,0]
    y = umap_embedding[0::down_samp,1]
    u = x[1::] - x[0:-1]
    v = y[1::] - y[0:-1]
     
    #calculate anlge of each arrow
    uv = np.vstack((u,v))
    theta = np.degrees(np.arctan2(*uv[::-1])) % 360.0
    norm =Normalize()
    norm.autoscale(theta)
    
    # use quiver to plot the arrows
    fig = plt.figure(dpi=200)
    #plt.quiver(x[0:-1], y[0:-1], u, v, color=cmap(theta))
    plt.quiver(x[0:-1], y[0:-1], u, v, color=cmap(norm(theta)))
    plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap))
    # plt.clim(0,360)
    plt.title(data_fn + 'quiver plot of umap embedding trajectories')
    fig.savefig(data_dir + data_fn + '_umap_traj_vector_field')
    
    # #aggregate all data for group plot
    # if data_fn == '201115_000':
    #     xx = x[0:-1]
    #     yy = y[0:-1]
    #     uu = u
    #     vv = v
    #     theta_all = theta
    # else: 
    #     xx = np.append(xx,x[0:-1])
    #     yy = np.append(yy,y[0:-1])
    #     uu = np.append(uu,u)
    #     vv = np.append(vv,v)
    #     theta_all = np.append(theta_all,theta)

#%% make aggregate vector field plot 

fig = plt.figure(dpi=300)
#plt.quiver(x[0:-1], y[0:-1], u, v, color=cmap(theta))
plt.quiver(xx, yy, uu, vv, color=cmap(norm(theta_all)), scale=400)
plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap))
# plt.clim(0,360)
plt.title('all mice quiver plot of umap embedding trajectories')

#%% use interpolation to creat a 2D field of vectors

# # make a meshgrid
# xx_min, yy_min = np.min(umap_embedding,axis=0)
# xx_max, yy_max = np.max(umap_embedding,axis=0)

# xx = np.linspace(np.floor(xx_min), np.ceil(xx_max), 100)
# yy = np.linspace(np.floor(yy_min), np.ceil(yy_max), 100)
# xx, yy = np.meshgrid(xx, yy)  

# #interpolate points over meshgrid
# points = np.transpose(np.vstack((x[0:-1], y[0:-1])))
# u_interp = interpolate.griddata(points, u, (xx, yy), method='linear')
# v_interp = interpolate.griddata(points, v, (xx, yy), method='linear')
# #xi=points at which to interpolate data

# #plotting 
# plt.figure(2, dpi=200)
# plt.quiver(xx, yy, u_interp, v_interp)
# plt.show()
