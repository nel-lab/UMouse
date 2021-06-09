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
import os

class UMouse:
    def __init__(self, expt_files=None, output_dir=None): 
        self.expt_files = []
        # # save mwt_paths as a class field
        # if isinstance(expt_files, str):
        #     expt_files = list([expt_files])
        # self.expt_files = expt_files
        
        # #save output dir as a class field
        # if output_dir is not None:
        #     if isinstance(output_dir, str):
        #         output_dir = list([output_dir])
        #     self.output_dir = output_dir

    def transform_mwt(self, df, f_sample, n_frequencies=25, fmin=1, fmax=None, columns=None):
        """
        Loads frames to generate the embedding fit and performs the fit. 

        Parameters
        ----------
        df : dataframe,  a path or list of paths
            Df, Path or list of paths for the behavioral data to be analyzed. The input format is compatible with DLC output
        n_frequencies : int
            number of groups to divide the frequencies range into.
        columns: list of strings
            Optional. In case only a subset of the columns need to be used
        fsample : int
            sampling frequency
        fmin : float
            minimum frequency of interest for the wavelet transformation.
        fmax : float
            maximum frequency of interest for the wavelet transformation.

        Returns
        -------
        Either a matrix, or a list of paths

        """
        # default to Nyquist frequency
        if fmax is None:
            fmax = 0.5*f_sample
            
        if type(df) is list:
            spect_paths = []
            freq_list = []
            power_list = []
            for ddf in df:
                df_data = pd.read_csv(ddf)
                
                if columns is not None:
                    df_data = df_data[[columns]]
                        
                freqs, power, spect_data = self._compute_mwt(df_data, n_frequencies, f_sample, fmin, fmax)
                
                output_path = ddf.split('.')[0]+'_mwt.npy'
                spect_paths.append(output_path)
                freq_list.append(freqs)
                power_list.append(power)
                
                np.save(output_path, spect_data)
                #spect_data.to_csv(path_or_buf = output_path, index=False)
                
            return freq_list, power_list, spect_paths
            
        else: 
            if type(df) is str:
                df_data = pd.read_csv(df)
            elif isinstance(df, pd.DataFrame): 
                df_data = df
            else:
                return print('Warning: input must be a path, list of paths, or dataFrame')
            
            if columns is not None:
                    df_data = df_data[[columns]]
            freqs, power, spect_data = self._compute_mwt(df_data, n_frequencies, f_sample, fmin, fmax)
        
            return freqs, power, spect_data 
    
    def fit_umap(self, data, fr_per_sess=None, **kwargs):
        
        """
        Loads mwt data to generate the umap embedding fit. Stores model as a class field.

        Parameters
        ----------
        data : npy array or list of paths
            Df, Path or list of both for the spectrographic data to be analyzed. The input format is compatible with DLC output

        **kwargs : dictionary
            keyword arguments for umap hyperparameters. See umap-learn.readthedocs.io

        Returns
        -------
        Either a matrix, or a list of paths
        
        Returns model as a class field

        """
        
        #initialize umap object
        self.UMAP = umap.UMAP(**kwargs)  
        
        if type(data) is list: # if input is a list of paths load each of them separately
            if fr_per_sess is None:
                fr_per_sess = np.round(50000/len(data))
                
            for dd in data:
                
                #check for file
                if os.path.exists(dd.split('.')[0]+'_mwt.npy'):
                    data_path = dd.split('.')[0]+'_mwt.npy'    
                elif os.path.exists(dd.split('.')[0]):    
                    data_path = dd.split('.')[0]
                else:
                    raise Exception('File:' + dd.split('.')[0]+'_mwt.npy' + 'does not exist, please run fit_mtw() before')
                
                #load and append precomputed spectrogrpahic data
                spect_data = np.load(data_path)  #spect_data of shape [frame, frequency]
                
                #collect sample frames from each mouse
                if 'fit_data' not in locals():
                    fit_data = spect_data[len(spect_data)//fr_per_sess,:]
                else:
                    fit_data = np.concatenate((fit_data, spect_data[len(spect_data)//fr_per_sess,:]))
 
        else:
            fit_data = data
            
        self.um_model = self.UMAP.fit(fit_data)           
            
    def transform_umap(self, transform_path, model=None):
        """
        Uses the embedding model generated by self.fit to transform the datasets indicated. 

        Parameters
        ----------
        transform_path : string or list
            Path or list of paths for the spectrographic data to be analyzed. 
            The default is to use all the datsets in self.pathnames.
        
        model : umap object
            Use a previously saved model to transform the data
        
        Returns
        -------
        None.

        """
        #If no model is input then use the model in the existing umap object
        if model is None:
            try: 
                model = self.UMAP
            except:
                raise Exception('No model found in UMAP object or specified as input parameter')
        
        #single mouse transform
        if len(transform_path) == 1 | isinstance(transform_path, str):
            
            #load the dataset
            if len(transform_path) == 1:
                spect_data = np.load(transform_path[0])  #spect_data of shape [frame, frequency]
            elif isinstance(transform_path, str):
                spect_data = np.load(transform_path)
            
            #transform the new dataset
            umap_embedding = model.transform(spect_data)
            
            #save the new embedding
            np.save(transform_path + '_embedding', umap_embedding)
            
        elif isinstance(transform_path, list):  
            #multi-mouse transform
            for this_path in transform_path:
                
                #load spectrographic data
                spect_data = np.load(this_path)
        
                #transform this dataset and save the embedding
                umap_embedding = model.transform(spect_data)
                
                #save the new embedding
                np.save(this_path + '_embedding', umap_embedding)
                    
                del spect_data
        else:
            raise Exception('input to transform_umap must be path(s) as a string or list of strings')
        
        

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
