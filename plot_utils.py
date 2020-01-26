#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:17:05 2020

@author: thatone

"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

sn.set(style = 'whitegrid', palette = 'muted', color_codes = True)
#plt.style.use('seaborn-whitegrid')
plt.style.use('fivethirtyeight')
figsize = (12, 10)

def plot_var_summary(data, var, hist_trunct_level = 1, max_cat = 10, save_path = None):
    """

    Parameters
    ----------
    data : pandas dataframe
    var : str
        column name
    hist_trunct_level : int [0:100]
         The default is 1. How many observations, probably outliers, should be truncated in %. 
         for example 5% means cut 5% of observations (2.5% from up and 2.5% from down)
    max_cat : int
        Maximum number of categories for categoric variable. Other will be aggregate to 'Other'. The default is 10.
    save_path : str, optional
         The default is None.

    Function takes data[var] and plots figures.
    If var numeric continous: histogram and na summary
    If var numeric and cardinality < 15: freq and summary
    if var char: freq and summary 

    """
    
    def truncate_histogram(data, var, hist_trunct_level):
        data_temp = data.dropna().sort_values(var).reset_index(drop = True)
        temp = data_temp.shape[0] * hist_trunct_level / 100
        temp_min = temp / 2
        temp_max = data_temp.shape[0] - temp / 2
        return data_temp.loc[temp_min : temp_max]
        
    data_ = pd.DataFrame()
    data_[var] = list(data[var])
    data_['nan']  = data_[var].isna()
    
    if hist_trunct_level != 0:
        data_hist_ = truncate_histogram(data = data_, var = var, hist_trunct_level = hist_trunct_level)
    else:
        data_hist_ = data
        
    # Check var type 
    if data_[var].dtype == np.object:
        v_type = 'o'
    elif data_[var].nunique() < 10:
        data_[var] = data_[var].astype('str') 
        v_type = 'o'
    else:
        v_type = 'n'
        
    if v_type == 'n':
        # Plot numeric variable 
        f, axes = plt.subplots(2, 2, figsize = figsize, sharex = False, constrained_layout = True)
        sn.despine(left = True)
        sn.distplot(data_hist_[var].dropna(), hist = True, ax = axes[0, 0], color = 'b')
        plt.xticks(rotation = 45)
        sn.distplot(data_hist_[var].dropna(), hist = True, ax = axes[0, 1], color = 'b', hist_kws = {'cumulative' : True}, kde_kws = {'cumulative' : True})
        sn.boxplot(data_[var], ax = axes[1, 0], color = 'b')
        axes[1,0].axvline(data_hist_[var].min(), color = 'gray', linestyle = '--')
        axes[1,0].axvline(data_hist_[var].max(), color = 'gray', linestyle = '--')
        print(data_['nan'].sum())
        sn.countplot(y = 'nan', data = data_, ax = axes[1, 1])   
        axes[0, 0].set_title('Histogram')
        axes[0, 1].set_title('Cumulative distribution')
        axes[1, 0].set_title('Box plot')
        axes[1, 1].set_title('Missing values')
        for ax in f.axes:
            plt.sca(ax)
            plt.xticks(rotation = 45)
        f.suptitle('Variable: ' + var)
        plt.tight_layout(rect = [0.05, 0, 1, 0.95])
        if save_path != None:
            plt.savefig(save_path)
        plt.show()
        plt.close()
        
    if v_type == 'o':
        # Plot categorical variable 
        def convert_categories(data):
            """
            
            Parameters
            ----------
            data : pandas dataframe data[var]

            Returns
            -------
            None.

            """
            counts = data.value_counts()    
            names = list(counts.index)
            values = list(counts)
            nan_values =  data.isna().sum()
            values_new = []
            names_new = []   
            cat_num = max_cat # Number of categories to display
            sum1 = 0
            if nan_values > 0:
                cat_num = cat_num - 1
            cat_num = cat_num - 1 
            for i in range(len(names)):
                if i < cat_num: 
                    values_new.append(values[i])
                    names_new.append(names[i])
                elif i >= cat_num:
                    sum1 = sum1 + values[i]
            if sum1 > 0:
                values_new.append(sum1)
                names_new.append('Other')
            if nan_values > 0:
                values_new.append(nan_values)
                names_new.append('Missing values')
            values_new, names_new = zip(*sorted(zip(values_new, names_new)))
            values_new = [i/sum(values_new)*100 for i in values_new]
            return values_new, names_new
        
        def get_colors(names):
            colors = []
            for i in range(len(names)):
                if names[i] == 'Other':
                    color = '#c44e52'
                elif names[i] == 'Missing values':
                    color = '#bb2a34'
                else:
                    color = '#2870b1'
                colors.append(color)
            return colors
            
        fig, ax = plt.subplots(figsize = figsize)
    
        values, names = convert_categories(data = data[var])
        y_pos = np.arange(len(values))
        ax.bar(y_pos, values, color = get_colors(names = names))
        ax.set_xticks(y_pos)
        ax.set_xticklabels(names, rotation = 75)
        plt.xlabel(var)
        plt.ylabel('%')
        fig.suptitle('Variable: ' + var)
        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path)
        plt.show()
        plt.close()
        

def plot_pearson_corr(data, plot_scatter = False, save_path = None): 
    """
    
    Plot correlations only for NUMERIC columns.

    Parameters
    ----------
    data : pandas df
        DESCRIPTION.
    plot_scatter : bool, optional
        If True then scatter plots and distributions will be shown.
    save_path : bool, optional
    
    Returns
    -------
    None.

    """       
    if not plot_scatter:
        corr = data.select_dtypes(exclude = ['object']).corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize = figsize)
        cmap = sn.diverging_palette(220, 10, as_cmap=True)
        sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Pearson corelation map')
        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path)
        plt.show()
        plt.close()
    else:
        sn.pairplot(data.select_dtypes(exclude = ['object']))
        plt.show()