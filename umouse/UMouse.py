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
    def __init__(self, n_frequencies=25, f_sample=70, fmin=1, fmax=None): 
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
        n_neighbors: int 
            see UMAP doc
        n_components: int
            see UMAP doc
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
        
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        
        
        #   fr_per_sess : integer, optional
        # Number of frames to sample from each dataset. Only used if multiple datasets are indicated in fit_path. 
        # The default is 50000/(n datasets).  

    def fit_mwt(self, df=None, columns=None):
        """
        Loads frames to generate the embedding fit and performs the fit. Stores model as a class field.

        Parameters
        ----------
        df : dataframe,  a path or list of paths
            Df, Path or list of both for the behavioral data to be analyzed. The input format is compatible with DLC output

        columns: list of strings
            in case only a subset of the columns need to be used

        Returns
        -------
        Either a matrix, or a list of paths

        """
        if type(df) is list:
           fit_data = [] 
           for ddf in df:
                df_one_mouse = pd.load_cvs(ddf)
                spect_data = self._compute_mwt(df_one_mouse, self.n_frequencies, self.f_sample, self.fmin, self.fmax)
                np.save(ddf.split('.')[0]+'_mwt.npy')
            else: 
                df_data = df
           
        else:
            if type(df) is str:
                df_data = pd.load_cvs(df)
            else: 
                df_data = df
                
            fit_data = self._compute_mwt(df_data, self.n_frequencies, self.f_sample, self.fmin, self.fmax)
        
        return fit_data 
        
    
    
    def fit_umap(data, fr_per_sess=None, n_neighbors=15, n_components=2,**kwargs):
        
        """
        Loads mwt data to generate the embedding fit. Stores model as a class field.

        Parameters
        ----------
        data : npy array or list of paths
            Df, Path or list of both for the spectrographic data to be analyzed. The input format is compatible with DLC output

      

        Returns
        -------
        Either a matrix, or a list of paths

        """
        
        #initialize umap object
        self.UMAP = umap.UMAP(,n_neighbors=15, n_components=2,**kwargs)  
        
        if type(data) is list: # if input is a list of pahs load each of them separately
            if fr_per_sess is None:
                fr_per_sess = np.round(50000/len(data))
                
            fit_data = []
            for dd in data:
                if os.path.exists(dd.split('.')[0]+'_mwt.npy'):                    
                    spect_data = np.load(dd.split('.')[0]+'_mwt.npy') # load precomputed spectrogrpahic data
                    fit_data.append(spect_data[:-1][::len(spect_data)//fr_per_sess]) # slect only fr_per_sess per mouse
                else:
                    raise Exception('File:' + dd.split('.')[0]+'_mwt.npy' + 'does not exist, please run fit_mtw() before')  
            fit_data = np.array(fit_data)
        else:
            fit_data=data
            
        self.um_model = self.UMAP.fit(fit_data)
        
        
            
        
        # #if subset of datasets are not specified for fitting then use all the available datasets
        # if fit_path is None:
        #     fit_path = self.mwt_paths
        
        # #single mouse fit
        # if len(fit_path) == 1 | isinstance(fit_path, str):
            
        #     #load the dataset
        #     if len(fit_path) == 1:
        #         fit_data = self.load_spect_data(fit_path[0])
        #     elif isinstance(fit_path, str):
        #         fit_data = self.load_spect_data(fit_path)
            
        #     #perform umap
        #     um_model = self.UMAP.fit(fit_data)
            
        #     #store the model
        #     self.model_ = um_model
            
        # else: #multi-mouse fit
        
        #     #determine number of frames to take from each session
        #     if fr_per_sess == None:
        #         fr_per_sess = np.round(50000/len(fit_path))
            
        #     #collect samples from each experiment to be used for a multi-mouse fitting
        #     for this_path in fit_path:
                
        #         spect_data = self.load_spect_data(this_path)
                
        #         #select the frames to sample
        #         n_frames = len(spect_data)
        #         samp_frames = np.round(np.linspace(0, n_frames, num=fr_per_sess, endpoint=False))
        #         samp_frames = list(samp_frames.astype(int))
        
        #         #collect sample frames from each mouse
        #         if this_path == fit_path[0]:
        #             multi_samp = spect_data[samp_frames,:]
        #         else:
        #             multi_samp = np.concatenate((multi_samp, spect_data[samp_frames,:]))
                    
        #         del spect_data
    
        #     #perform umap embedding
        #     um_model = self.UMAP.fit(multi_samp)
            
    def transform(self, transform_path=None):
        """
        Uses the embedding model generated by self.fit to transform the datasets  indicated. 

        Parameters
        ----------
        transform_path : string or list, optional
            Path or list of paths for the spectrographic data to be analyzed. 
            The default is to use all the datsets in self.pathnames.

        Returns
        -------
        None.

        """
        #if subset of datasets are not specified to transform then transform all the available datasets
        if transform_path is None:
            transform_path = self.pathnames
            
        #single mouse transform
        if len(transform_path) == 1 | isinstance(transform_path, str):
            
            #load the dataset
            if len(transform_path) == 1:
                spect_data = self.load_spect_data(transform_path[0])
            elif isinstance(fit_path, str):
                spect_data = self.load_spect_data(transform_path)
            
            #transform the new dataset
            this_embedding = self.UMAP.transform(spect_data)
            
            #save the new embedding
            np.savetxt(self.output_dir + self.filenames + '_embedding.csv', this_embedding, delimiter=",")
            
        else:  
            #multi-mouse transform
            for this_path in transform_path:
                
                #load spectrographic data
                spect_data = self.load_spect_data(this_path)
        
                #transform this dataset and save the embedding
                this_embedding = self.UMAP.transform(spect_data)
                
                #save the new embedding
                this_filename =  os.path.basename(this_path)
                
                np.savetxt(self.output_dir + this_filename + '_embedding.csv', 
                           this_embedding,
                           delimiter=",")
                    
                del spect_data
    
        
        

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
