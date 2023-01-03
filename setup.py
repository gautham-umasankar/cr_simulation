#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 17:35:02 2022

@author: gautham
"""

from setuptools import setup

setup(name='cr_simulation',
      version='1.0',
      description='Package for numerical simulation of CR and single qubit gates',
      url='https://github.com/gautham-umasankar/cr_simulation',
      author='Gautham Umasankar',
      author_email='gautham.umasankar@yale.edu',
      packages=['cr_simulation'],
      install_requires = ['qutip==4.7.1', 'tqdm>=4.64.1', 'notebook', 'matplotlib', 'joblib', 'sympy', 'tabulate'],
      zip_safe=False)
