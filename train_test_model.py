#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:54:29 2020

@author: thatone
"""
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from linear_regression_fs import LinearRegressionFS

DATA_DIR = './data/'
GRP_PATH = './grp/'

def train_test():
    
    # Read data
    X_train = pd.read_csv(DATA_DIR + 'x_train.csv')
    y_train = pd.read_csv(DATA_DIR + 'y_train.csv')
    X_test = pd.read_csv(DATA_DIR + 'x_test.csv')
    y_test = pd.read_csv(DATA_DIR + 'y_test.csv')
    
    # Train model1 
    model1 = LinearRegressionFS(fit_intercept = True)
    model1.fit(X = X_train, y = y_train, save_path = GRP_PATH + 'model1_') 
    model1_train_results  = model1.diagnose(X_train, y_train, save_path = GRP_PATH + 'model1_train_')
    model1.show_statsmodels_result(X_train, y_train, save_path = GRP_PATH + 'model1_summ.png')
        
    # Test model1
    model1_test_results = model1.diagnose(X_test, y_test, save_path = GRP_PATH + 'model1_test_')
            
    # Add PolynomialFeatures
    poly = PolynomialFeatures(2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train model2    
    model2 = LinearRegressionFS(fit_intercept = True)
    model2.fit(X = X_train_poly, y = y_train, save_path = GRP_PATH + 'model2_') 
    model2_train_results  = model2.diagnose(X_train_poly, y_train, save_path = GRP_PATH + 'model2_train_')
    model2.show_statsmodels_result(X = X_train_poly, y = y_train, save_path = GRP_PATH + 'model2_summ.png')
        
    # Test model2
    model2_test_results = model2.diagnose(X_test_poly, y_test, save_path = GRP_PATH + 'model2_test_')
    
    # Compare results
    slow = {'model1_train' : model1_train_results, 'model1_test' : model1_test_results, 'model2_train': model2_train_results, 'model2_test' : model2_test_results}
    LinearRegressionFS.compare_results(slow)
    
if __name__ == '__main__':
    train_test()