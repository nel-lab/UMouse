# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:38:04 2020

To be run after open_and_format_RW_data.py

Script for plotting performing Morlet Wavelet Transformation of Richard Warren's
locomotion data. Data involves a headfixed animal on a running wheel with an occasional
obstacle and reward. 

@author: Jake
"""

#make a list of datasets
# data_fn_list  = list(['181215_003', '201115_000', 
#                      '201217_000', '201218_000', '201226_000', '201227_000', 
#                      '201228_000', '201229_000', '201230_000'])

data_fn_list  = list(['201115_000', 
                     '201217_000', '201218_000', '201226_000', '201227_000', 
                     '201228_000', '201229_000', '201230_000'])

#%% import dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

#%% access the data
for data_fn in data_fn_list: 
    
    #set path
    data_dir = 'D:/data/BehaviorData/RW_data/' + data_fn + '/'
    print(data_fn)
    
    trackingDf = pd.read_csv(data_dir + data_fn + '_Df.csv') 
    pawsRS = genfromtxt(data_dir + data_fn + '_pawsArray.csv', delimiter=',') 
    
    #%% Morlet wavelet analysis
    
    frame_rate = np.round(1/np.mean(np.diff(trackingDf['timeStamps'][0:3000])))
    n_freqs = 25  #np.round(frame_rate/10)
    mwt_input = np.concatenate((pawsRS, 
                              trackingDf['bodyAngles'].to_numpy().reshape(len(pawsRS),1), 
                              trackingDf['jawVarX'].to_numpy().reshape(len(pawsRS),1)), 
                            axis=1)
    
    #load module and perform transformation
    from behavelet import wavelet_transform
    print('starting Morlet Wavelet Transform')
    freqs, power, X_new = wavelet_transform(mwt_input, 
                                            n_freqs=n_freqs, 
                                            fsample=frame_rate, 
                                            fmin=1., 
                                            fmax=50.)
    
    #save variables for use later
    print('Morlet Wavelet Transform complete. Saving variables.')
    np.savetxt(data_dir + data_fn + '_mwtFreqs.csv', freqs, delimiter=",")
    np.savetxt(data_dir + data_fn + '_mwtPower.csv', power, delimiter=",")
    np.savetxt(data_dir + data_fn + '_mwtXNew.csv', X_new[:,0:(n_freqs*pawsRS.shape[1])], 
               delimiter=",")
    
    #save variables with jaw angles and body angles for use later
    np.savetxt(data_dir + data_fn + '_jawBod_mwtXNew.csv', X_new, delimiter=",")
    
    #%% OPTIONAL - shortcut if you have already performed behavelet
    if 'X_new' not in locals():
        
        #without jaw angle and body angle
        #X_new = genfromtxt(data_dir + data_fn + '_mwtXNew.csv', delimiter=',')
        
        #with jaw angle and body angle
        X_new = genfromtxt(data_dir + data_fn + '_jawBod_mwtXNew.csv', delimiter=',')
    
    #%%  plot the behavelet data
        
    #plot spectropgrahic data with vertical lines indicating whisker contact and reward
    wiskArray = np.array(np.where(trackingDf['wiskContTimeBool'])[0])
    rewardArray = np.array(np.where(trackingDf['rewardBool'])[0])
    
    fig = plt.figure()
    ax1 = fig.add_axes([0,0,1,1])
    ax1.imshow(X_new[:50000,0:350].T, aspect='auto')
    #Whisker contact times (only plotting the first frame of contact sequence)
    for wiskInd,thisWisk in enumerate(wiskArray):
        if wiskInd==0:
            ax1.axvline(x=thisWisk, color = 'w', lw=0.5)
        elif thisWisk > 50000:
            break
        elif thisWisk - wiskArray[wiskInd-1]>1:
            ax1.axvline(x=thisWisk, color = 'w', lw=0.5)
    #Reward
    for rewInd,thisRew in enumerate(rewardArray):
        if rewInd==0:
            ax1.axvline(x=thisRew, color = 'b', lw=0.5)
        elif thisRew > 50000:
            break
        elif thisRew - rewardArray[rewInd-1]>1:
            ax1.axvline(x=thisRew, color = 'b', lw=0.5)
         
    #for plotting without the indactor lines
    # plt.imshow(X_new[:50000,:].T, aspect='auto')
    # #plt.imshow(X_new.values[:50000,:].T, aspect='auto')
    # plt.axvline(x=np.where(trackingDf['wiskContTimeBool'][:50000]), color='w')
    
    plt.title('Behavelet output ' + data_fn + ' white=whisker, blue=reward')
    plt.ylabel('Paws * dimensions')
    plt.xlabel('frame # at 250Hz')
    plt.savefig(data_dir + data_fn + 'spectImg')
    
    #%% plot whisk contact time triggered averages
    
    plotWiskWindow = np.array(range(-125,250))
    if 'wiskArray' not in locals():
        wiskArray = np.array(np.where(trackingDf['wiskContTimeBool'])[0])
    
    #find all frame #s of the first whisk in a whisker contact sequence
    for wiskInd,thisWisk in enumerate(wiskArray):
        if wiskInd==0:
            all_wisk = X_new[plotWiskWindow + thisWisk,:]
        elif thisWisk >= len(X_new)-250:
            break
        elif thisWisk - wiskArray[wiskInd-1]>1 and len(all_wisk.shape)==2:
            all_wisk = np.stack((all_wisk, X_new[plotWiskWindow + thisWisk,:]))
            all_wisk = np.reshape(all_wisk, (375,350,2))
        elif thisWisk - wiskArray[wiskInd-1]>1:
            all_wisk = np.dstack((all_wisk, X_new[plotWiskWindow + thisWisk,:]))
    
    fig = plt.figure()
    ax1 = fig.add_axes([0,0,1,1])
    ax1.imshow(np.mean(all_wisk[:,0:300,:], axis=2).T, aspect='auto')
    ax1.axvline(x=-plotWiskWindow[0], color = 'w', lw=0.5)
    
    plt.savefig(data_dir + data_fn + 'whiskTrigAvgSpectro')
    
    # if X_new.shape[1] > 300:
    #     X_new = X_new[:,0:300]
            

