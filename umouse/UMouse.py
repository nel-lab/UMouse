#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:02:56 2021

@author: andreagiovannucci
"""
import numpy as np
import pandas as pd
from behavelet import wavelet_transform
import umap

class UMouse:
    def __init__(self, n_frequencies=25, f_sample=70, fmin=1, fmax=None): #, **kwargs
        """
        
        Parameters
        ----------
        n_frequencies : int
            number of groups to divide the frequencies range into.
        fsample : int
            sampling frequency
        fmin : float
            minimum frequency of interest for the wavelet transformation.
        fmax : float
            maximum frequency of interest for the wavelet transformation.
        
        (OLD)
        mwt_paths : string or list
            path for the spectrographic dataset to be analyzed.
        output_dir : string
            destination for analysis outputs. If no destination is specified they will be saved to pathname
        **kwargs : Misc
            keyword arguments for the UMAP class object initialization

        Returns
        -------
        None.

        """
        
        # default to Nyquist frequency
        if fmax is None:
            fmax = 0.5*f_sample
    
        self.n_frequencies=n_frequencies
        self.f_sample=f_sample
        self.fmin=fmin
        self.fmax=fmax
        
        # # save mwt_paths as a class field
        # if isinstance(mwt_paths, str):
        #     mwt_paths = list([mwt_paths])
        # self.mwt_paths = mwt_paths
        
        # #get the filename for experiemnt
        # self.filenames = list(map(os.path.basename, mwt_paths))
        
        # #save output dir as a class field
        # if output_dir is None:
        #     self.output_dir = list(map(os.path.dirname, self.mwt_paths))
        # else:
        #     if isinstance(output_dir, str):
        #         output_dir = list([output_dir])
        #     self.output_dir = output_dir
        
        #initialize umap object
        #self.UMAP = umap.UMAP(**kwargs)
        
        
    def _compute_mwt(self, behavior_df, n_frequencies, f_sample, fmin, fmax):
        """
        Perform morlet wavelett transformation on the DLC data

        Parameters
        ----------
        behavior_df : pandas dataframe
            dataframe containing the DLC trajectory data.
        n_frequencies : int
            number of groups to divide the frequencies range into.
        fsample : int
            sampling frequency
        fmin : float
            minimum frequency of interest for the wavelet transformation.
        fmax : float
            maximum frequency of interest for the wavelet transformation.

        Returns
        -------
        freqs : ndarray, shape (n_freqs)
            The frequencies used for the wavelet transform
        power : ndarray, shape (n_samples)
            The total power for each row in X_new
        mwt_array : numpy array, shape (n_samples, n_features*n_freqs)
            Continuous wavelet transformed data

        """
        
        #perform transformation
        freqs, power, mwt_array = wavelet_transform(behavior_df.to_numpy(), 
                                                    n_freqs=n_frequencies, 
                                                    fsample=f_sample, 
                                                    fmin=fmin, 
                                                    fmax=fmax)
        
        return freqs, power, mwt_array