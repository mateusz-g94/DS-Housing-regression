#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:25:25 2020

@author: thatone
"""
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from pandas.plotting import table

sn.set(style = 'whitegrid', palette = 'muted', color_codes = True)
plt.style.use('seaborn-whitegrid')
plt.style.use('fivethirtyeight')
figsize = (12, 10)

class LinearRegressionFS(LinearRegression):
    """
    Linear regression with feature selection.
    """  
    def _select_features(self, X, y, best, save_path, method = 'rfecv', min_features = 1):
        if best:
            if method == 'rfecv':
                print('Select best')
                rfecv = RFECV(estimator = LinearRegression(self.fit_intercept), min_features_to_select = min_features, cv = KFold(3), scoring = 'neg_mean_squared_error', n_jobs = -1)
                rfecv.fit_transform(X = X, y = y)
                self.bf_support_ = rfecv.support_
                self.bf_n_features_ = rfecv.n_features_
                # Results
                print("Optimal number of features : %d" % rfecv.n_features_)
                # Plot number of features VS. cross-validation scores
                plt.figure(figsize = figsize)
                plt.xlabel("Number of features selected")
                plt.ylabel("Cross validation score (neg mean squared error)")
                plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
                if save_path != None:
                    plt.savefig(save_path + 'rfecv_feature_selection.png')
                plt.show()
            if method == 'chi2':
                pass
        else: 
            self.bf_support_ = [True] * X.shape[1]
            
    def _set_columns(self, X):
        if isinstance(X, np.ndarray):
            self.column_names_ = None
        elif isinstance(X, pd.DataFrame):
            self.column_names_ = [v[0] for v in zip(X.columns, self.bf_support_) if v[1]]
        else:
            raise TypeError('Unsuported data structure.')
    
    def _select_columns(self, X):
        if isinstance(X, np.ndarray):
            return X[:, self.bf_support_]
        elif isinstance(X, pd.DataFrame):
            return X[self.column_names_]
        else:
            raise TypeError('Unsuported data structure.')
    
    @staticmethod        
    def plot_text_to_png(data, path, figsize = (12,10)):
        fig, ax = plt.subplots(figsize = figsize)
        if isinstance(data, pd.DataFrame):
            table(ax, data, loc = 'upper left')
        else:
            plt.text(0.01, 0.05, str(data), {'fontsize': 10}, fontproperties = 'monospace')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path)
    
    @staticmethod    
    def compare_results(results = {}, save_path = None):
        df = pd.DataFrame()
        MSE = []
        R2 = []
        names = []
        n_features = []
        for slow in results:
            MSE.append(results[slow]['MSE'])
            R2.append(results[slow]['R2'])
            names.append(slow)
            n_features.append(results[slow]['n_features'])
        df['names'] = names
        df['MSE'] = MSE
        df['R2'] = R2
        df['n_features'] = n_features
        print(df)

    def fit(self, X, y, best = True, selection_method = 'rfecv', min_features = 1, save_path = None):
        self._select_features(X = X, y = y, best = best, method = selection_method, min_features = min_features, save_path = save_path)
        self._set_columns(X = X)
        X_ = self._select_columns(X)
        super().fit(X = X_, y = y)
        
    def get_coefs(self):      
        if isinstance(self.column_names_, type(None)):
            column_names = [v[0] for v in zip([i for i in range(1,len(self.bf_support_) + 1) ], self.bf_support_) if v[1]]
        else:
            column_names = self.column_names_
        coefs = dict(zip(column_names, list(self.coef_[0])))
        return coefs
    
    def predict(self, X):
        X_ = self._select_columns(X)
        return super().predict(X = X_)
        
    def diagnose(self, X, y, save_path = None):
        y_pred = self.predict(X)
        resid = [v[0] - v[1] for v in zip(y.values, y_pred)]
        resid = [r[0] for r in resid]
        coefs = self.get_coefs()
        MSE = mean_squared_error(y, y_pred)
        R2 = round(self.score(X, y),3)
        shapiro_stat = stats.shapiro(resid)       
        try:
            intercept = self.intercept_[0]
        except TypeError:
            intercept = self.intercept_     
            
        # Diagnostic plots
        fig, ax = plt.subplots(figsize = figsize)
        sn.distplot(resid, fit = stats.norm, kde = False)
        plt.xlabel('residuals')
        plt.ylabel('%')
        plt.title('Histogram of residuals and fitted normal distribution')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path + 'residuals_hist.png')
        plt.show()
        
        fig, ax = plt.subplots(figsize = figsize)
        plt.scatter(y, resid)
        
        plt.xlabel('y')
        plt.ylabel('residuals')
        plt.title('Scatter plot residuals vs y')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path + 'residuals_vs_y.png')
        plt.show()
        
        fig, ax = plt.subplots(figsize = figsize)
        plt.scatter(y, y_pred)
        
        plt.xlabel('y')
        plt.ylabel('prediction')
        plt.title('Scatter plot y vs prediction')
        plt.plot([min(y_pred), max(y.values)],[min(y_pred), max(y.values)], 'k-', alpha = 0.5, zorder = 0)
        plt.xticks(rotation = 45)
        plt.xlim(min(y_pred), max(y.values))
        plt.ylim(min(y_pred), max(y.values))
        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path + 'y_vs_prediction.png')
        plt.show()
        
        return {'coefs': coefs, 'intercept' : intercept, 'MSE' : MSE, 'R2' : R2, 'n_features' : self.bf_n_features_, 'vars' : self.column_names_, 'shapiro_stat' : shapiro_stat}
        
    def show_statsmodels_result(self, X, y, save_path = None):
        X_ = self._select_columns(X)
        if self.fit_intercept:
            X_ = sm.add_constant(X_)
        sm_model = sm.OLS(y, X_).fit()
        return sm_model
        print(sm_model.summary())
        if save_path != None:
            self.plot_text_to_png(data = sm_model.summary(), path = save_path)
        
    
    
        