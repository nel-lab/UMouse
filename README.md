# UMouse video based behavior analysis
Repo for analyzing and visualizing DLC data. The package uses a python implementation of Uniform Manifold Approximation and Projection (UMAP). The package performs the following operations:
- Compute wavelet transform (it uses the [behavelet package](https://pypi.org/project/behavelet/)
- Compute the UMAP transform of data (from the UMAP-learn package [UMAP](https://umap-learn.readthedocs.io/en/latest/) )
- Visualize a low-dimensional representation of behavior and highlights behavioral events
UMouse has been tested on Linux Ubuntu and MacOS. First the user must install the dependencies as outlined in the installation instructions. UMouse is run using the Python via an Anaconda distribution. 

# Installation
- Install [Anaconda](https://www.anaconda.com/products/individual)
- clone the [UMouse directory](https://github.com/nel-lab/UMouse.git)
- change directory to UMouse repo ```cd [path/to/local/repo]```
- run ```conda env create -f environment.yml -n UMouse```
- run ```conda activate UMouse```
- run ```pip install -e.```
- run ```pip install behavelet```

# Usage

## 1. ```UMouse.py```
> The UMouse class object will allow you to create a 2 or 3 dimensional UMAP embedding from your data. It takes as input 3D coordinates for multiple tracked body part locations. Input data should be in a Pandas DataFrame with columns corresponding to the X, Y, Z and coordinates for each tracked body part over time. See the demo section below for a link to an example dataset. 
> 1. Initialize UMouse class object and set hyperparameters for Morlet Wavelet Transform and UMAP embedding. 
> 2. Run the ```.fit()``` method within the UMouse class object with a list of filenames you wish to analyze. This step will perform a Morlet wavelet transformation (MWT) and save the output. 
> 3. Run the ```.transform()``` method within the UMouse class object while passing in a list of dataset filenames you wish to transform from a MWT data into a low dimensional UMAP embedding. 

## 2. ```UMousePlotter_functions.py```
> This script contains multiple plotting functions and a class object for visualizing the UMAP embedding of your data. The interactive features may require the matplotlib backend set to ```%matplotlib```
> 
> Functions:  
> plot_embedding() Make a scatterplot of the UMAP embedding.  
> plot_continuous_var() Make a scatterplot with color axis representing a continuous variable such as locomotion velocity.   
> vector_field_plot() Make a quiver plot in which arrows indicate the local dynamics within the UMAP embedding.  
> plot_categorical_var() Make scatterplot where color axis represents categorical variables such as reward times.  
> get_points_high_dim() Enter coordinates of points to plot traces/movies if the UMAP embedding is > 2 dimensions  
> play() Play the montage movie created by interactive.behavior_montage()  
> 
> Class Object interactive() with methods:  
> get_points() Select points from the UMAP embedding to plot either DLC traces or raw movies  
> plot_traces() Plot DLC traces from selected limbs in windows around the points selected in get_points()  
> behavior_montage() Plot raw movies from selected limbs in windows around the points selected in get_points()  

# Demo
> A demo is available in ```use_cases/UMouse_demo.py``` The demo uses datasets accessible on [Dropbox](https://www.dropbox.com/sh/sn1ru8sf19icb4u/AAA7Q70qVq2XwVSMywmG0FOpa?dl=0)
> Instructions for how to run the demo can be found in each code block. It is best run in blocks within the Spyder IDE. 

# Developers
- William Heffley, UNC
- Jimmy Tabet, UNC
- Andrea Giovannucci, UNC
 
