# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:16:51 2020

t-sne development script 

@author: Jake
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import cv2

# Set path
data_dir = 'D:/data/Behavior data/RW_data/'

#%% OPTIONAL - shortcut if you have already performed behavelet

X_new = pd.read_csv('D:/data/Behavior data/RW_data/X_newArray.csv')
trackingDf = pd.read_csv('D:/data/Behavior data/RW_data/trackingDf.csv')
pawsRS = genfromtxt('D:/data/Behavior data/RW_data/trackingArray.csv', delimiter=',')

#%% Run tsne on all data points 

comp = 3
t = TSNE(n_components=comp-1)
tsne = t.fit_transform(X_new)

#%% calculate kde from tsne 2dim data



#%% visualize kde plot

