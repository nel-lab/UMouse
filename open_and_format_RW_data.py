# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:48:18 2020

Open Warren data, explore variables. Reformat timestamped data to booleans.
Clean the data. Convert to dataframe and save file.

@author: Jake Heffley
"""

#make a list of datasets
data_fn_list  = list(['trackingData_181215_003'], ['trackingData_201115_000'], 
                     ['201217_000'], ['201218_000'], ['201226_000'], ['201227_000'], 
                     ['201228_000'], ['201229_000'], ['201230_000'])

#select the dataset to analyze
#data_fn = 'trackingData_201115_000'
data_fn = 'trackingData_181215_003'
expt_fn = data_fn[-10:]


#%% import dependencies
import pandas as pd
import numpy as np
from scipy.io import loadmat

#%%  unpack dctionary

# open the data. In dictionary format. 
mat_data = loadmat('D:/data/BehaviorData/RW_data/' + data_fn + '/' + data_fn + '.mat')

# keys: obstacleTimes, paws, paws_pixels, rewardTimes, t,  etc
# Values are numpy arrays
list(mat_data.keys())

#check the shape of all the variabels  #shape for 201115 = (403912, 4, 3)
# np.shape(mat_data['t']) # = (317617, 1) # t converted to tVar. Frame timestamps in seconds.
# np.shape(mat_data['obstacleTimes']) # = (179, 2)
# np.shape(mat_data['paws']) # = (317617, 4, 3) #3D array with dimensions  time, paw#, and axis (XYZ)
# np.shape(mat_data['paws_pixels']) # = (317617, 4, 3)
# np.shape(mat_data['rewardTimes']) # = (61, 1) #inter reward interval varies ~16-25s
# mat_data['vel'].shape #(frame#, 1)
# mat_data['whiskerAngle'].shape  #(frame#, 1)
# mat_data['bodyAngles'].shape #(frame#, 1)
# mat_data['jaw'].shape  #(frame#, 2) #range = 100:127. ~15k NaNs

# obstacle duration varies ~1-3s in duration
tVar = mat_data['t']
obstTimes = mat_data['obstacleTimes'] 
paws = mat_data['paws']
pawsPixels = mat_data['paws_pixels']
rewardTimes = mat_data['rewardTimes']
velVar = mat_data['vel']
whiskerAngle = mat_data['whiskerAngle']
bodyAngles = mat_data['bodyAngles']
jawVar = mat_data['jaw']

# additional keys not included in 181215_003 dataset
if 'lickTimes' in list(mat_data.keys()):
    mat_data['lickTimes'].shape #(#licks, 1)  vals=timestamps in seconds
    lickTimes = mat_data['lickTimes']
if 'wiskContactTimes' in list(mat_data.keys()):
    mat_data['wiskContactTimes'].shape #(1, Xcontacts) vals=timestamps in seconds
    wiskContactTimes = mat_data['wiskContactTimes']

#%% data cleaning - look for nan values in the data and impute or remove

if '181215_003' in data_fn: 
    #remove first 807 [0:807] timepoints due to NaNs in the Z axis for FL, FR, BR paws
    #need to drop frames from paws, pawsPixels, tVar, velVar, bodyAngles, whiskerAngle, jawVar
    f_cut = 807
    paws = paws[f_cut:]
    pawsPixels = pawsPixels[f_cut:]
    tVar = tVar[f_cut:]
    velVar = velVar[f_cut:]
    bodyAngles = bodyAngles[f_cut:]
    whiskerAngle = whiskerAngle[f_cut:]
    jawVar = jawVar[f_cut:]

#Look for nan values in the tracking data
good_frames = np.sum(np.sum(paws, axis=2), axis=1)
assert np.sum(np.isnan(good_frames)) == 0
assert np.sum(np.isnan(bodyAngles)) == 0
assert np.sum(np.isnan(np.sum(jawVar, axis=1))) == 0
assert np.sum(np.isnan(velVar)) == 0
assert np.sum(np.isnan(whiskerAngle)) == 0
assert np.sum(np.isnan(tVar)) == 0
assert np.sum(np.isnan(np.sum(obstTimes, axis=1))) == 0

#Deal with nans in jaw and wiskContactTimes variables
if 'wiskContactTimes' in list(mat_data.keys()):
    if np.sum(np.isnan(wiskContactTimes)) != 0:
        wiskContactTimes = wiskContactTimes[~np.isnan(wiskContactTimes)]
if 'lickTimes' in list(mat_data.keys()):
    if np.sum(np.isnan(lickTimes)) != 0:
        lickTimes = lickTimes[~np.isnan(lickTimes)]
if np.sum(np.isnan(rewardTimes)) != 0:
    lickTimes = lickTimes[~np.isnan(rewardTimes)]
    
#trim lickTimes so it only includes values with associated DLC coordinates (paws)
if 'lickTimes' in list(mat_data.keys()):
    lickTimes = lickTimes[lickTimes > np.min(tVar)+0.05]
    lickTimes = lickTimes[lickTimes < np.max(tVar)-0.05]
#do the same for wiskContactTimes
if 'wiskContactTimes' in list(mat_data.keys()):
    wiskContactTimes = wiskContactTimes[wiskContactTimes > np.min(tVar)+0.05]
    wiskContactTimes = wiskContactTimes[wiskContactTimes < np.max(tVar)-0.05]

#%% check imaging rate. Should be 250Hz

frame_rate = 1/np.diff(mat_data['t'][0:1000].T).mean()
assert np.round(frame_rate) == 250

#%% create boolean versions of rewardTimes, obstTimes, lickTimes, and wiskContactTimes

#Create boolean with 1s for the timepoints in a 0.5s window after reward
rewTimeBool = np.zeros(len(paws))
for thisRew in range(0, len(rewardTimes)): 
    postRewIx = np.argwhere((tVar > rewardTimes[thisRew]) & (tVar < rewardTimes[thisRew]+0.5)) 
    rewTimeBool[postRewIx[:,0]] = 1  

#Create a boolean dataframe column with 1s during the obstacle
obsTimeBool = np.zeros(len(paws))
for thisObs in range(0, len(obstTimes)): 
    obsIx = np.argwhere((tVar > obstTimes[thisObs,0]) & (tVar < obstTimes[thisObs,1])) 
    obsTimeBool[obsIx[:,0]] = 1 

#create a boolean for lickTimes. With times points -50ms:+50ms around lick == 1
if 'lickTimes' in list(mat_data.keys()):
    lickTimeBool = np.zeros(len(paws))
    for thisLick in range(0, len(lickTimes)): 
        obsIx = np.argwhere((tVar > lickTimes[thisLick]-0.50) & (tVar < lickTimes[thisLick]+0.05)) 
        lickTimeBool[obsIx[:,0]] = 1 

#create boolean for interval 250ms after wiskContactTimes 
#Paper indicates around 30 ms for movement based reaction to whiskers touch bar
if 'wiskContactTimes' in list(mat_data.keys()):
    wiskContTimeBool = np.zeros(len(paws))
    for thisWisk in range(0, len(wiskContactTimes)): 
        postWiskIx = np.argwhere((tVar > wiskContactTimes[thisWisk]) & (tVar < wiskContactTimes[thisWisk]+0.25)) 
        wiskContTimeBool[postWiskIx[:,0]] = 1  

#%% Convert variables into a pandas dataframes

#generate labels for each paw and axis: front/back, left/right, XYZ
dfCols = ['BLX', 'BLY', 'BLZ', 'FLX', 'FLY', 'FLZ', 'FRX', 'FRY', 'FRZ', 'BRX', 'BRY', 'BRZ',]
pawsRS = np.reshape(paws, [paws.shape[0], paws.shape[1]*paws.shape[2]])

#create dataframe and add variables
trackingDf = pd.DataFrame(data = pawsRS, columns=dfCols)
trackingDf['timeStamps'] = tVar
trackingDf['jawVarX'] = jawVar[:,0]
trackingDf['jawVarY'] = jawVar[:,1]
trackingDf['velVar'] = velVar
trackingDf['whiskerAngle'] = whiskerAngle
trackingDf['bodyAngles'] = bodyAngles

#store the boolean arrays in the dataframe
trackingDf['rewardBool'] = rewTimeBool
trackingDf['obstacleBool'] = obsTimeBool
if 'lickTimes' in list(mat_data.keys()):
    trackingDf['lickTimeBool'] = lickTimeBool
if 'wiskContactTimes' in list(mat_data.keys()):
    trackingDf['wiskContTimeBool'] = wiskContTimeBool

#%% Save dataframe and the array 
trackingDf.to_csv(path_or_buf = 'D:/data/BehaviorData/RW_data/' + data_fn + '/' + expt_fn + '_Df.csv', 
                  index=False)
np.savetxt('D:/data/BehaviorData/RW_data/' + data_fn + '/' + expt_fn + '_pawsArray.csv', 
           pawsRS, 
           delimiter=",")




