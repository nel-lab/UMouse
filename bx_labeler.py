# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:03:29 2021

Script for making behavior labels to use in umap plotting
To be run after the behavelet transformation and before umap/mouseLeap

@author: Jake
"""

#make a list of datasets
data_fn_list  = list(['181215_003', '201115_000', 
                     '201217_000', '201218_000', '201226_000', '201227_000', 
                     '201228_000', '201229_000', '201230_000'])

#select the dataset to analyze
data_fn = data_fn_list[8]
print(data_fn)

#set path
data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'

#%% import dependencies

import pandas as pd
import numpy as np

#%% load data

trackingDf = pd.read_csv(data_dir + data_fn + '_Df.csv') 

#%% Separate obstacle times into early and late

obstDiff = np.diff(trackingDf['obstacleBool'])

obstStart = np.where(obstDiff==1)[0] + 1 #add 1 to adjust index for np.diff
obstEnd = np.where(obstDiff==-1)[0] + 1

obstEarly = np.zeros([1,len(trackingDf['obstacleBool'])])
obstMid   = np.zeros([1,len(trackingDf['obstacleBool'])])
obstLate  = np.zeros([1,len(trackingDf['obstacleBool'])])

n_div = 3

#Make separate indeces for early, middle, and late obstacle times
for bout in range(0, len(obstStart)):
    assert obstStart[bout] < obstEnd[bout]
    
    obstDur = obstEnd[bout] - obstStart[bout]
    
    #make sure that the duration is divisible by the number of groups. 
    if obstDur % 3 != 0: #trim off the front end of the duration index until no remainder
        obstInd = np.array(range(obstStart[bout] + (obstDur % 3), obstEnd[bout])) 
    else:
        obstInd = np.array(range(obstStart[bout], obstEnd[bout]))
    
    splitInds = np.split(obstInd, n_div)
    
    #hardcoded this bit for 3 groups but could be moded for any n groups
    obstEarly[0][splitInds[0]] = 1
    obstMid[0][splitInds[1]] = 1
    obstLate[0][splitInds[2]] = 1 

#Make behavior label for un-downsampled data point
# 1=reward  2= early obstacle   3 = mid obstacle   4 = late obstacle
bx_labels = np.zeros([1,len(trackingDf)])
bx_labels[0, [np.where(trackingDf.rewardBool ==1)]] = 1
bx_labels[0, [np.where(obstEarly.T ==1)]] = 2
bx_labels[0, [np.where(obstMid.T ==1)]] = 3
bx_labels[0, [np.where(obstLate.T ==1)]] = 4

np.savetxt(data_dir + data_fn + "_bxLabelsArray.csv", 
           bx_labels, 
           delimiter=",")


