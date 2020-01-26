#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:47:05 2020

@author: thatone

"""
import pandas as pd
import os
import datetime
from sklearn.model_selection import ShuffleSplit

DATA_DIR = './data/'
TARGET_NAME = 'price'

def prepare_train_data():
    """
    Prepare data and save file to DATA_DIR

    Returns
    -------
    None.

    """
    data = pd.read_csv(DATA_DIR + 'house_data.csv')
    for col in data.columns:
        print(col, ' : ', data[col].dtype)
        
    # Transform columns
    # data['date_diff'] = [(datetime.date.today() - datetime.date(int(dt[0:4]), int(dt[4:6]), int(dt[6:8]))).days for dt in data['date']]
    data['yr_built_diff'] = [datetime.date.today().year - yr_built for yr_built in data['yr_built']]
    data['f_renovated'] = [1 if yr_ren > 0 else 0 for yr_ren in data['yr_renovated']]
    data['zipcode'] = data['zipcode'].astype(str)
    # Drop redundant columns
    data.drop(['id', 'date', 'yr_built', 'yr_renovated'], axis = 1, inplace = True)
    
    # Check zipcodes
    data['zipcode'].value_counts() # High cardinality, this variable wont be a good predictor, we cant aggregate categories or one hot encode
    data.drop(['zipcode'], axis = 1, inplace = True) # The same information - lat, long
    data.drop(['lat', 'long'], axis = 1, inplace = True) # Uninformative, not linear 
    data.drop(['waterfront'], axis = 1, inplace = True) # low cardinality
    
    # save prepared data to file
    data.to_csv(DATA_DIR + 'house_data_pre.csv', index = False)
    
    # Split into X and y 
    x_cols = [col for col in data.columns if col != TARGET_NAME]
    data_x = data[x_cols]
    data_y = data[[TARGET_NAME]]
    
    # Split into test and train
    spl = ShuffleSplit(n_splits = 2, test_size = 0.2, random_state = 7)
    for train_index, test_index in spl.split(data_x, data_y):
        x_train, x_test = data_x.iloc[train_index], data_x.iloc[test_index]
        y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        
    # Save files
    x_train.to_csv(os.path.join(DATA_DIR, 'x_train.csv'), index = False)    
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index = False) 
    x_test.to_csv(os.path.join(DATA_DIR, 'x_test.csv'), index = False) 
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index = False) 
    
if __name__ == '__main__':
    prepare_train_data()

