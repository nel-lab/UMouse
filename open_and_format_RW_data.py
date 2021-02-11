# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:48:18 2020

@author: Jake
"""

# Open Warren data, explore variables, Identify the indeces of specific paws. 
#reformat reward and obstacle data to times series, convert to dataframe and save file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# open the data. In dictionary format. 
mat_data = loadmat('D:/data/Behavior data/RW_data/trackingData_181215_003.mat')

#%%  unpack dctionary

# keys: obstacleTimes, paws, paws_pixels, rewardTimes, t
# Values are numpy arrays
list(mat_data.keys())

# t converted to t_var. Frame timestamps in seconds.
np.shape(mat_data['t']) # = (317617, 1)
tVar = mat_data['t'];
#np.diff(t_var[0:15], axis=0) #timestamps at roughly 250Hz

#not sure what is on the 2nd dim. Perh
np.shape(mat_data['obstacleTimes']) # = (179, 2)
obstTimes = mat_data['obstacleTimes'] #obstacle duration varies ~1-3s in duration

#3D array with dimensions  time, paw#, and ???
np.shape(mat_data['paws']) # = (317617, 4, 3)
paws = mat_data['paws'];

np.shape(mat_data['paws_pixels']) # = (317617, 4, 3)
pawsPixels = mat_data['paws_pixels'];

np.shape(mat_data['rewardTimes']) # = (61, 1)
rewardTimes = mat_data['rewardTimes']; #inter reward interval varies ~16-25s

#%% Convert variables into a pandas dataframes

dfCols = ['BLX', 'BLY', 'BLZ', 'FLX', 'FLY', 'FLZ', 'FRX', 'FRY', 'FRZ', 'BRX', 'BRY', 'BRZ',]
pawsRS = np.reshape(paws, [317617, 12])

trackingDf = pd.DataFrame(data = pawsRS, columns=dfCols)
trackingDf['timeStamps'] = tVar

#Make boolean dataframe column with 1s for the timepoints in a window after reward
rewTimeBool = np.zeros(len(pawsRS))
for thisRew in range(0, len(rewardTimes)): 
    postRewIx = np.argwhere((tVar > rewardTimes[thisRew]) & (tVar < rewardTimes[thisRew]+0.5)) # 0.5s post reward
    rewTimeBool[[postRewIx[:,0]]] = 1  
#store the reward array in the dataframe
trackingDf['rewardBool'] = rewTimeBool

#Create a boolean dataframe column with 1s during the obstacle
#perhaps divide up obstacle times into first half and second half of each obstacle motion so as
#to capture front paw then back paw. 
obsTimeBool = np.zeros(len(pawsRS))
for thisObs in range(0, len(obstTimes)): 
    obsIx = np.argwhere((tVar > obstTimes[thisObs,0]) & (tVar < obstTimes[thisObs,1])) # 0.5s post reward
    obsTimeBool[[obsIx[:,0]]] = 1  

trackingDf['obstacleBool'] = obsTimeBool

#%% data cleaning
#remove first 807 [0:806] timepoints due to NaNs in the Z axis for FL, FR, BR paws
trackingDf.drop(trackingDf.index[0:807], inplace = True)
pawsRS = pawsRS[807:]

#%% Save dataframe and the array 
trackingDf.to_csv(path_or_buf = 'D:/data/Behavior data/RW_data/trackingDf.csv', index=False)
np.savetxt("D:/data/Behavior data/RW_data/trackingArray.csv", pawsRS, delimiter=",")

#%% Plot values for paws_pixels
# Seems like dimension 0 is the x value in pixels, dim1 is Ybot, dim2 is YTop. 
# There is a mirro so the paw is in two positions on the y axis 
# and one on the x axis. Range[0:375?]

#paws IDs 0=BL   1=FL   2=FR   3=BR   Assumes pixel 0,0 is top left
fig = plt.figure()
plt.hist(paws_pixels[:,0,0], bins=40, color='r', alpha=0.7)
plt.hist(paws_pixels[:,0,1], bins=40, color='g', alpha=0.7)
plt.hist(paws_pixels[:,0,2], bins=40, color='b', alpha=0.7)

fig = plt.figure()
plt.hist(paws_pixels[:,1,0], bins=40, color='r', alpha=0.7)
plt.hist(paws_pixels[:,1,1], bins=40, color='g', alpha=0.7)
plt.hist(paws_pixels[:,1,2], bins=40, color='b', alpha=0.7)

fig = plt.figure()
plt.hist(paws_pixels[:,2,0], bins=40, color='r', alpha=0.7)
plt.hist(paws_pixels[:,2,1], bins=40, color='g', alpha=0.7)
plt.hist(paws_pixels[:,2,2], bins=40, color='b', alpha=0.7)

fig = plt.figure()
plt.hist(paws_pixels[:,3,0], bins=40, color='r', alpha=0.7)
plt.hist(paws_pixels[:,3,1], bins=40, color='g', alpha=0.7)
plt.hist(paws_pixels[:,3,2], bins=40, color='b', alpha=0.7)

#%% plot values for paws
#range [-0.06:0.10]
fig = plt.figure()
plt.hist(paws[:,0,0], bins=40, color='r', alpha=0.7)
plt.hist(paws[:,0,1], bins=40, color='g', alpha=0.7)
plt.hist(paws[:,0,2], bins=40, color='b', alpha=0.7)


#%% Determine ID of specific paws and their index in the matrices
#pawsPixels.shape (317617, 4, 3)
print('pawsPixels variable')
print('paw 0: ', np.nanmean(pawsPixels[:,0,:], axis=0))
print('paw 1: ', np.nanmean(pawsPixels[:,1,:], axis=0))
print('paw 2: ', np.nanmean(pawsPixels[:,2,:], axis=0))
print('paw 3: ', np.nanmean(pawsPixels[:,3,:], axis=0))

#same for paws
print('paws variable')
print('paw 0: ', np.nanmean(paws[:,0,:]*100, axis=0))
print('paw 1: ', np.nanmean(paws[:,1,:]*100, axis=0))
print('paw 2: ', np.nanmean(paws[:,2,:]*100, axis=0))
print('paw 3: ', np.nanmean(paws[:,3,:]*100, axis=0))

#plot paw positions in video
fig = plt.figure()
plt.scatter()

#index and paw ID
#IDs 0=BL   1=FL   2=FR   3=BR   Assumes pixel 0,0 is top left
#XYZ order in index vals is  []
