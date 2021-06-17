# -*- coding: utf-8 -*-
"""


@author: William Heffley
"""
 
import numpy as np
import pandas as pd
from scipy.io import loadmat

def WarrenDataProcess(expt_pathname, output_path=None, paws_list=None, 
                 lick_window=None, whisk_react_window=None, reward_window=None):
    """
    Specific to the data collected by Richard Warren's open source locomotion setup. 
    load the matlab data set, clean the data, and convert to a pandas dataframe.
    
    Parameters
    ----------
    expt_pathname : string
        pathname of the dataset to process.
    output_path : string
        destination for analysis outputs. If no destination is specified they will be saved to pathname

    Returns
    -------
    behavior_df : dataframe shape (n_samples, n_cols)
        Pandas dataframe containing the dlc coordinates and behavior event variables.

    """
    #save output dir as a class field
    if output_path is None:
        output_path = expt_pathname.split('.')[0] + '_behavior_df'
            
    #identify the paws to be tracked and label them Front/Back, Left,Right, X/Y/Z
    if paws_list == None:
        paws_list = ['BLX', 'BLY', 'BLZ', 
                          'FLX', 'FLY', 'FLZ', 
                          'FRX', 'FRY', 'FRZ', 
                          'BRX', 'BRY', 'BRZ']
    
    #default value for analysis windows
    if lick_window is None:
        lick_window = 0.05
    if whisk_react_window is None:
        whisk_react_window = 0.25
    if reward_window is None:
        reward_window = 1.0
    
    # open the data. In dictionary format. 
    mat_data = loadmat(expt_pathname)
    
    #check for nans at the beginning vs end of paws
    good_frames = np.sum(np.sum(mat_data['paws'], axis=2), axis=1)
    if np.sum(np.isnan(good_frames[0:1000])) != 0:
        early_cut = np.sum(np.isnan(good_frames[0:1000]))
    else:
        early_cut = 0
    if np.sum(np.isnan(good_frames[-1000:])) != 0:
        late_cut = len(good_frames)-np.sum(np.isnan(good_frames[-1000:]))
    else:
        late_cut = len(good_frames)
    
    #remove frames from beginning vs end of paws, t, bodyAngles, jaw, vel, whsikerAngle
    mat_data['paws'] = mat_data['paws'][early_cut:late_cut,:,:]
    mat_data['t'] = mat_data['t'][early_cut:late_cut]
    mat_data['bodyAngles'] = mat_data['bodyAngles'][early_cut:late_cut]
    mat_data['jaw'] = mat_data['jaw'][early_cut:late_cut,:]
    mat_data['vel'] = mat_data['vel'][early_cut:late_cut,:]
    mat_data['whiskerAngle'] = mat_data['whiskerAngle'][early_cut:late_cut,:]
    
    #check for values outside of t within  wiskContactTimes, lickTimes, obstacleTimes, rewardTimes 
    t_min = mat_data['t'][0]
    t_max = mat_data['t'][-1]
    mat_data['wiskContactTimes'] = mat_data['wiskContactTimes'][np.logical_and((mat_data['wiskContactTimes']>t_min), (mat_data['wiskContactTimes']<t_max))]
    mat_data['lickTimes'] = mat_data['lickTimes'][np.logical_and((mat_data['lickTimes']>t_min), (mat_data['lickTimes']<t_max))]
    mat_data['rewardTimes'] = mat_data['rewardTimes'][np.logical_and((mat_data['rewardTimes']>t_min), (mat_data['rewardTimes']<t_max))]
    mat_data['obstacleTimes'] = mat_data['obstacleTimes'][np.logical_and((mat_data['obstacleTimes'][:,0]>t_min), (mat_data['obstacleTimes'][:,1]<t_max)), :]
    
    #check for nans in wiskContactTimes
    if np.sum(np.isnan(mat_data['wiskContactTimes'])) != 0:
        mat_data['wiskContactTimes'] = mat_data['wiskContactTimes'][~np.isnan(mat_data['wiskContactTimes'])]
    
    # make variables for each column which is used
    tVar = mat_data['t']
    obstTimes = mat_data['obstacleTimes'] 
    paws = mat_data['paws']
    rewardTimes = mat_data['rewardTimes']
    velVar = mat_data['vel']
    whiskerAngle = mat_data['whiskerAngle']
    bodyAngles = mat_data['bodyAngles']
    jawVar = mat_data['jaw']
    lickTimes = mat_data['lickTimes']
    wiskContactTimes = mat_data['wiskContactTimes']
    
    #Look for nan values in the tracking data
    good_frames = np.sum(np.sum(paws, axis=2), axis=1)
    assert np.sum(np.isnan(good_frames)) == 0
    assert np.sum(np.isnan(bodyAngles)) == 0
    assert np.sum(np.isnan(np.sum(jawVar, axis=1))) == 0
    assert np.sum(np.isnan(velVar)) == 0
    assert np.sum(np.isnan(whiskerAngle)) == 0
    assert np.sum(np.isnan(tVar)) == 0
    assert np.sum(np.isnan(np.sum(obstTimes, axis=1))) == 0
    
    #create boolean versions of rewardTimes, obstTimes, lickTimes, and wiskContactTimes

    #Create boolean with 1s for the timepoints in a 1.0s window after reward
    rewTimeBool = np.zeros(len(paws))
    for thisRew in range(0, len(rewardTimes)): 
        postRewIx = np.argwhere((tVar > rewardTimes[thisRew]) & (tVar < rewardTimes[thisRew]+reward_window)) 
        rewTimeBool[postRewIx[:,0]] = 1  
    
    #Create a boolean dataframe column with 1s during the obstacle
    obsTimeBool = np.zeros(len(paws))
    for thisObs in range(0, len(obstTimes)): 
        obsIx = np.argwhere((tVar > obstTimes[thisObs,0]) & (tVar < obstTimes[thisObs,1])) 
        obsTimeBool[obsIx[:,0]] = 1 
    
    #create a boolean for lickTimes. With times points -X ms : +X ms around lick == 1
    if 'lickTimes' in list(mat_data.keys()):
        lickTimeBool = np.zeros(len(paws))
        for thisLick in range(0, len(lickTimes)): 
            obsIx = np.argwhere((tVar > lickTimes[thisLick]-lick_window) & (tVar < lickTimes[thisLick]+lick_window)) 
            lickTimeBool[obsIx[:,0]] = 1 
    
    #create boolean for interval 250ms after wiskContactTimes 
    #~30 ms for movement based reaction to whiskers touch bar
    if 'wiskContactTimes' in list(mat_data.keys()):
        wiskContTimeBool = np.zeros(len(paws))
        for thisWisk in range(0, len(wiskContactTimes)): 
            postWiskIx = np.argwhere((tVar > wiskContactTimes[thisWisk]) & (tVar < wiskContactTimes[thisWisk]+whisk_react_window)) 
            wiskContTimeBool[postWiskIx[:,0]] = 1  
    
    # Convert variables into a pandas dataframes

    #generate labels for each paw and axis: front/back, left/right, XYZ
    paws = np.reshape(paws, [paws.shape[0], paws.shape[1]*paws.shape[2]])
    
    #create dataframe and add variables
    behavior_df = pd.DataFrame(data = paws, columns=paws_list)
    behavior_df['timeStamps'] = tVar
    behavior_df['jawVarX'] = jawVar[:,0]
    behavior_df['jawVarY'] = jawVar[:,1]
    behavior_df['velVar'] = velVar
    behavior_df['whiskerAngle'] = whiskerAngle
    behavior_df['bodyAngles'] = bodyAngles
    
    #store the boolean arrays in the dataframe
    behavior_df['rewardBool'] = rewTimeBool
    behavior_df['obstacleBool'] = obsTimeBool
    if 'lickTimes' in list(mat_data.keys()):
        behavior_df['lickTimeBool'] = lickTimeBool
    if 'wiskContactTimes' in list(mat_data.keys()):
        behavior_df['wiskContTimeBool'] = wiskContTimeBool
    
    #save the behavior df
    behavior_df.to_csv(path_or_buf = output_path, index=False)
    
    return behavior_df
    
        
        
        




