#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:25:35 2021

@author: jake
"""
from setuptools import setup, find_packages

setup(
      name='UMouse',
      version='1.0',
      packages=find_packages(include=['UMouse', 'Umouse.*']),
      install_requires=['']
      )

