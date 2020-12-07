# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:47:36 2020

Need to get data by running sections of behavelet_and_decomp.py 

To be a repository for various plots used to describe decomposition of behavelet locomotion 
data from Richard Warren. 

@author: Jake
"""


#%% 2 dim plotting for tsne. Max iter=1000
# 0=other, 1=reward+500ms, 2=obst1/3,   3=obstMid,   4=obstEnd
scores_plot = tsne
fig = plt.figure()
ax = plt.axes(title='2dim tsne cyan=reward, obstacle=gbk, red=other')

ax.scatter(*scores_plot.T[:,[np.where(PC_labels==0)[1]]], c='r', marker='o', alpha=0.05) #other
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==1)[1]]], c='c', marker='o', alpha=0.2) #reward
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==2)[1]]], c='g', marker='o', alpha=0.2) #obst1/3
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==3)[1]]], c='b', marker='o', alpha=0.2)
ax.scatter(*scores_plot.T[:,[np.where(PC_labels==4)[1]]], c='k', marker='o', alpha=0.2)

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')

#%% some experimental code to save the 3d plots of decomposition outputs. 
#Sadly, pickeling does not preserve the ability to manipulate viewing angle 
#  for the 3d matplotlib plots. 

import pickle

#pickle the figure to save it for later
pickle_out = open("FigureObject.fig.pickle","wb")
pickle.dump(fig, pickle_out)
pickle_out.close()

#code to open a pickled figure
pickle_in = open("FigureObject.fig.pickle","rb")
figx = pickle.load(pickle_in)


