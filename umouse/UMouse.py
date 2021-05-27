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
    def __init__(self, params): #, **kwargs
        """
        
        Parameters
        ----------
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
        # save mwt_paths as a class field
        if isinstance(mwt_paths, str):
            mwt_paths = list([mwt_paths])
        self.mwt_paths = mwt_paths
        
        #get the filename for experiemnt
        self.filenames = list(map(os.path.basename, mwt_paths))
        
        #save output dir as a class field
        if output_dir is None:
            self.output_dir = list(map(os.path.dirname, self.mwt_paths))
        else:
            if isinstance(output_dir, str):
                output_dir = list([output_dir])
            self.output_dir = output_dir
        
        #initialize umap object
        #self.UMAP = umap.UMAP(**kwargs)
        
        
        