#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:06:17 2020

@author: thatone

1) Plot vars distrib 
2) Plot correlation 
3) Deal with missing data
4) Transform data


"""

# from DataScience.plot_utilities import plot_var_summary
from plot_utils import plot_var_summary, plot_pearson_corr
import pandas as pd

DATA_DIR = ',/data/'
GRP_PATH = './grp/'

data = pd.read_csv(DATA_DIR + 'house_data_pre.csv', index_col = False)

# Plot variables distribution
for col in data.columns:
    path_ex = GRP_PATH + col + '.png'
    print(path_ex)
    plot_var_summary(data = data, var = col, hist_trunct_level = 5, max_cat = 6, save_path = path_ex)
    
# Plot correlation map
path_ex = GRP_PATH + 'corr.png'
plot_pearson_corr(data = data, save_path = path_ex)


