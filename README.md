# UMouse video based behavior analysis
Repo for analyzing and visualizing DLC data. The package performs the following operations:
- Compute wavelet transform (it uses the behavelet package [:INK]
- Compute the UMAP transform of data 
- Visualize a low-dimensional representation of behavior and highlights behavioral events

# installation
- Install Anaconda
- type ```conda env create -f environment.yml -n UMouse```
- type ```pip install behavelet```

# Usage

### 1. """UMouse.py"""
The UMouse class object will allow you to create a 2 or 3 dimensional UMAP embedding from your data. It takes as input 3D coordinates for multiple tracked body part locations. 

### 2. """UMousePlotter_functions.py"""
This script contains multiple plotting functions and a class object for visualizing the UMAP embedding of your data.

# Demo
A demo is available in """use_cases/UMouse_demo.py"""
The demo uses datasets accessible on Dropbox at:
"""https://www.dropbox.com/sh/sn1ru8sf19icb4u/AAA7Q70qVq2XwVSMywmG0FOpa?dl=0"""

# developers
- William Heffley, UNC
- Jimmy Tabet, UNC
- Andrea Giovannucci, UNC
 
