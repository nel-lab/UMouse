# -*- coding: utf-8 -*-
"""

Refactored code for the UMouse pipeline
Loading and preprocessing class

Take data from Richard Warren's locomotion setup.
Open, preprocess, and convert to dataframe.

inputs pathnames for behavior files 
@author: William Heffley
"""
 
import os
import numpy as np
import pandas as pd
import behavelet
from scipy.io import loadmat

class UMouseLoader:
    
    
    def __init__(self, expt_pathname, output_dir=None, paws_list=None):
        """
        
        Parameters
        ----------

        expt_pathname : string
            path for the dataset to be analyzed.
        output_dir : string
            destination for analysis outputs. If no destination is specified they will be saved to pathname

        Returns
        -------
        None.

        """
        
        #get the filename for experiemnt
        self.filename = os.path.basename(expt_pathname)
        
        # save pathnames as a class field
        self.pathname = expt_pathname
        
        #save output dir as a class field
        if output_dir is None:
            self.output_dir = self.pathname
        else:
            self.output_dir = output_dir
        
        #identify the paws to be tracked and label them Front/Back, Left,Right, X/Y/Z
        if paws_list == None:
            self.paws_list = ['BLX', 'BLY', 'BLZ', 
                              'FLX', 'FLY', 'FLZ', 
                              'FRX', 'FRY', 'FRZ', 
                              'BRX', 'BRY', 'BRZ']
        else:
            self.paws_list = paws_list
        
    def load_data(self, expt_pathname, lick_window=None, 
                  whisk_react_window=None, reward_window=None):
        """
        load the matlab data set, clean the data, and convert to a pandas dataframe
        
        Parameters
        ----------
        expt_pathname : string
            pathname of the dataset to process.

        Returns
        -------
        behavior_df : dataframe shape (n_samples, n_cols)
            Pandas dataframe containing the dlc coordinates and behavior variables.

        """
        
        #default value for analysis windows
        if lick_window is None:
            lick_window = 0.05
        if whisk_react_window is None:
            whisk_react_window = 0.25
        if reward_window is None:
            reward_window = 1.0
        
        # open the data. In dictionary format. 
        mat_data = loadmat(expt_pathname)
        
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
        behavior_df = pd.DataFrame(data = paws, columns=self.paws_list)
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
        
        return behavior_df
        
    def mwt(self, behavior_df, n_frequencies=None, fmin=None, fmax=None,
            bodyAngle=True, jawAngle=True):
        """
        Perform morlet wavelett transformation on the DLC data

        Parameters
        ----------
        behavior_df : pandas dataframe
            dataframe containing the DLC trajectory data.
        n_frequencies : int
            number of groups to divide the frequencies range into.
        fmin : float
            minimum frequency of interest for the wavelet transformation.
        fmax : float
            maximum frequency of interest for the wavelet transformation..

        Returns
        -------
        freqs : ndarray, shape (n_freqs)
            The frequencies used for the wavelet transform
        power : ndarray, shape (n_samples)
            The total power for each row in X_new
        mwt_df : pandas nd dataframe, shape (n_samples, n_features*n_freqs)
            Continuous wavelet transformed data

        """
        
        #default value for Morlet Wavelet Transformation
        if n_frequencies is None:
            n_frequencies = 25
        if fmin is None:
            fmin = 1.
        if fmax is None:
            fmax = 50.
        frame_rate = np.round(1/np.mean(np.diff(behavior_df['timeStamps'][0:1000])))
            
        # construct input array for MWT
        mwt_input = behavior_df[self.paws_list]
        mwt_cols = self.paws_list
        
        if bodyAngle:
            np.concatenate(mwt_input,
                            behavior_df['bodyAngles'].to_numpy().reshape(len(behavior_df),1),
                            axis=1)
            mwt_cols.append('bodyAngles')
        
        if jawAngle:
            np.concatenate(mwt_input,
                            behavior_df.iloc[:['jawVarX', 'jawVarY']].to_numpy().reshape(len(behavior_df),2),
                            axis=1)   
            mwt_cols.append('jawVarX', 'jawVarY')
        
        #perform transformation
        freqs, power, X_new = wavelet_transform(mwt_input, 
                                                n_freqs=n_frequencies, 
                                                fsample=frame_rate, 
                                                fmin=fmin, 
                                                fmax=fmax)
        
        #transform MWT data into dataframe
        mwt_df = pd.DataFrame(data = X_new, columns=mwt_cols)
        
        return freqs, power, mwt_df
    
    
    def label_behavior(self, behavior_df):
        """
        label the behavioral timepoints according to the obstacle and reward periods

        Parameters
        ----------
        behavior_df : dataframe
            Pandas dataframe containing the dlc coordinates and behavior variables.

        Returns
        -------
        bx_labels : ndarray shape(1,n_samples)
            vector with coded values for non-overlapping events in each trial. 
            1=reward  2= early obstacle   3 = mid obstacle   4 = late obstacle
            

        """
        
        # Separate obstacle times into early and late
        
        obstDiff = np.diff(behavior_df['obstacleBool'])
        
        obstStart = np.where(obstDiff==1)[0] + 1 #add 1 to adjust index for np.diff
        obstEnd = np.where(obstDiff==-1)[0] + 1
        
        obstEarly = np.zeros([1,len(behavior_df['obstacleBool'])])
        obstMid   = np.zeros([1,len(behavior_df['obstacleBool'])])
        obstLate  = np.zeros([1,len(behavior_df['obstacleBool'])])
        
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
        bx_labels = np.zeros([1,len(behavior_df)])
        bx_labels[0, [np.where(behavior_df.rewardBool ==1)]] = 1
        bx_labels[0, [np.where(obstEarly.T ==1)]] = 2
        bx_labels[0, [np.where(obstMid.T ==1)]] = 3
        bx_labels[0, [np.where(obstLate.T ==1)]] = 4
        
        return bx_labels
        
    def load_mwt_label(self, expt_pathname, saveVars=False):
        """
        performs all three transformations on each data set listed in the pathnames variable

        Parameters
        ----------
        expt_pathname : string
            pathname of the dataset to process.

        Returns
        -------
        behavior_df : dataframe shape (n_samples, n_cols)
            Pandas dataframe containing the dlc coordinates and behavior variables.
        freqs : ndarray, shape (n_freqs)
            The frequencies used for the wavelet transform
        power : ndarray, shape (n_samples)
            The total power for each row in X_new
        mwt_df : pandas nd dataframe, shape (n_samples, n_features*n_freqs)
            Continuous wavelet transformed data
        bx_labels : ndarray shape(1,n_samples)
            vector with coded values for non-overlapping events in each trial. 
            1=reward  2= early obstacle   3 = mid obstacle   4 = late obstacle

        """
        
        behavior_df = self.load_data(expt_pathname)
        
        freqs, power, mwt_df = self.mwt(behavior_df, bodyAngle=True, jawAngle=True)
        
        bx_labels = self.label_behavior(behavior_df)
        
        if saveVars == True:
            self.save_outputs(self, behavior_df=behavior_df, freqs=freqs, power=power, mwt_df=mwt_df, bx_labels=bx_labels)
        
        return behavior_df, freqs, power, mwt_df, bx_labels
    
    def save_outputs(self, behavior_df=None, freqs=None, power=None, mwt_df=None, bx_labels=None):
        """
        Optionally saves all output variables produced by UMouseLoader

        Parameters
        ----------
        behavior_df : dataframe shape (n_samples, n_cols)
            Pandas dataframe containing the dlc coordinates and behavior variables.
        freqs : ndarray, shape (n_freqs)
            The frequencies used for the wavelet transform
        power : ndarray, shape (n_samples)
            The total power for each row in X_new
        mwt_df : pandas nd dataframe, shape (n_samples, n_features*n_freqs)
            Continuous wavelet transformed data
        bx_labels : ndarray shape(1,n_samples)
            vector with coded values for non-overlapping events in each trial. 
            1=reward  2= early obstacle   3 = mid obstacle   4 = late obstacle

        Returns
        -------
        None.

        """
        
        try: 
            behavior_df.to_csv(path_or_buf = self.output_dir + self.filename + '_behavior_df.csv', index=False)
        except:
               print('error while saving behavior_df') 
    
        try:
            np.savetxt(self.output_dir + self.filename + '_freqsArray.csv', freqs, delimiter=",")
        except:
               print('error while saving freqs') 
        
        try:
            np.savetxt(self.output_dir + self.filename + '_powerArray.csv', power, delimiter=",")
        except:
               print('error while saving power') 
        
        try: 
            mwt_df.to_csv(path_or_buf = self.output_dir + self.filename + '_mwt_df.csv', index=False)
        except:
               print('error while saving mwt_df') 
        
        try:
            np.savetxt(self.output_dir + self.filename + "_bxLabelsArray.csv", bx_labels, delimiter=",")
        except:
               print('error while saving bx_labels') 
        
        
        