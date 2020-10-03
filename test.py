#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:21:34 2020

@author: samirrachidzaim-admin
"""

# evaluate random forest ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree
import pandas as pd
import scipy.stats as st
import statsmodels.stats.multitest as correct

# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=12, noise=0.1, random_state= 2)
# define the model
model = RandomForestRegressor(max_depth=2,  n_estimators=20, max_features=5)
# evaluate the model
rf = model.fit(X,y)
# report performance



def getMainEffectCounts(rf):
    ## get root splitting var 
    features= [[tree.tree_.feature[0]] for j,tree in enumerate(rf.estimators_)]
    df = pd.Series(data=features)

    ## calculate frequency of forot split vars
    df = df.value_counts().rename_axis('feature').reset_index(name='TestStatistic')
    return(df)

def calculatePvalue(df): 
    df = pd.Series(data=features)
    df = df.value_counts().rename_axis('feature').reset_index(name='TestStatistic')
    df['pvalues'] = [st.binom_test(x, 20, 1/20, alternative='greater') for x in df.TestStatistic]
    fdrs = correct.fdrcorrection(df.pvalues,  method='negcorr' )[1]
    df['FDR']= fdrs
    return(df)

def plotPvalues(df):
    

'''
This stack overflow page contains info for how to access 
individual trees and decision paths :
    
    https://stackoverflow.com/questions/48869343/decision-path-for-a-random-forest-classifier
    


'''

