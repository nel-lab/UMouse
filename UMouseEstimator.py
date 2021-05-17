# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:23:56 2021

Refactored code for the UMouse pipeline
umap fitting and embedding class

take preprocessed behavior data from UMouseLoader and perform the UMAP embedding

@author: Jake
"""
# need fit and fit_transform to be able to pass args to UMAP


class UMouseEstimator:
    def __init__(self, pathnames, output_dir, **kwargs):
        """
        

        Parameters
        ----------
        pathnames : TYPE
            DESCRIPTION.
        output_dir : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        tot_embed_fr = 50000
        n_sess = len(data_fn_list8)
        fr_per_sess = int(np.round(tot_embed_fr/n_sess)) 
        
        #get the filename for experiemnt
        self.filenames = os.path.basename(pathnames) 
        
        # save pathnames as a class field
        self.pathnames = pathnames
        
        #save output dir as a class field
        self.output_dir = output_dir
        
        #initialize umap object
        self.UMAP = umap.UMAP(**kwargs)
        
    def load_spect_data(expt_pathname):
        
        #load spectrographic data
        this_spect = pd.read_csv(expt_pathname, header=None)
        this_spect = this_spect.to_numpy()
        
    def fit(fit_path=None, fr_per_sess=None):
        """
        

        Parameters
        ----------
        fit_path : TYPE, optional
            DESCRIPTION. The default is None.
        fr_per_sess : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        #if subset of datasets are not specified for fitting then use all the available datasets
        if fit_path is None:
            fit_path = self.pathnames
        
        #single mouse fit
        if len(fit_path) == 1 | isinstance(fit_path, str):
            
            #load the dataset
            if len(fit_path) == 1:
                fit_data = self.load_spect_data(fit_path[0])
            elif isinstance(fit_path, str):
                fit_data = self.load_spect_data(fit_path)
            
            #perform umap
            um_model = self.UMAP.fit(fit_data)
            
            #store the model
            self.model_ = um_model
            
        else:     
        
            #determine number of frames to take from each session
            if fr_per_sess=None:
                fr_per_sess = np.round(50000/len(fit_path))
            
            #multi-mouse fit
            for this_path in fit_path:
                
                spect_data = self.load_spect_data(this_path)
                
                #select the frames to sample
                n_frames = len(spect_data)
                samp_frames = np.round(np.linspace(0, n_frames, num=fr_per_sess, endpoint=False))
                samp_frames = list(samp_frames.astype(int))
        
                #collect sample frames from each mouse
                if this_path == fit_path[0]:
                    multi_samp = spect_data[samp_frames,:]
                else:
                    multi_samp = np.concatenate((multi_samp, spect_data[samp_frames,:]))
                    
                del spect_data
    
            #perform umap embedding
            um_model = self.UMAP.fit(multi_samp)
            
    def transform(transform_path=None):
        """
        

        Parameters
        ----------
        transform_path : TYPE, optional
            DESCRIPTION. The default is None.

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
            np.savetxt(self.output_dir + self.filenames + '_embedding.csv', 
            this_embedding,
            delimiter=",")
            
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
            
    def fit_transform(fit_path=None, transform_data=None):
        
        #if subset of datasets are not specified for fitting then use all the available datasets
        if fit_path is None:
            fit_path = self.pathnames
            
        #if subset of datasets are not specified to transform then transform all the available datasets
        if transform_path is None:
            transform_path = self.pathnames
            

        

