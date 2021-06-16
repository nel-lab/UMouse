#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:27:22 2021

@author: jake
"""
import numpy as np
import pandas as pd
import os

import time


#%% 
#os.chdir([Umouse repo path])

from umouse.UMouse import UMouse
#from umouse.UMousePlotter import UMousePlotter

#os.chdir([data path])

#%%  #Load and preprocess example data from Richard Warren's open source treadmill setup 



file_list = ['201229_000',
             '201230_000'
    ]
# '201115_000', '201217_000','201218_000', '201226_000', '201227_000',



# %%
expt_pathname = 'D:/data/BehaviorData/RW_data/201218_000/201218_000_behavior_df'


for filename in file_list:
    expt_pathname = 'trackingData_' + filename +'.mat'
    output_dir = 'UMouseOutput/' + filename + '_behavior_df'
    

#%%
#    n_frequencies=25, f_sample=70, fmin=1, fmax=None, n_neighbors=15, n_components=2,**kwargs
um_estimate = UMouse(n_frequencies=25, f_sample=250, fmin=1, fmax=None, n_neighbors=15, n_components=2)
columns_list = 'BLX', 'BLY', 'BLZ', 'FLX', 'FLY', 'FLZ', 'FRX', 'FRY', 'FRZ', 'BRX', 'BRY', 'BRZ'
f_sample=250
#%% Run morlet wavelet transform to get spectrographic data
    
#   fr_per_sess=5000, df=None, columns=None, recompute_mwt=False, n_neighbors=None, n_components=None

start = time.time()

df = ['201229_000_behavior_df']
fit_data = um_estimate.fit(df, fr_per_sess=80000, columns=columns_list)

end = time.time()
print(end - start)

#%%  

transform_path = []
um_estimate.transform(transform_path)









