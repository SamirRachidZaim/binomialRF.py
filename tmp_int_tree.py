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


#### access each decision tree
rf.estimators_[0]
sample_id = 0

tree_dict = dict()
leaves = dict()
features = dict()

trees = dict()

for j, tree in enumerate(rf.estimators_):
    #print(tree)

    trees[j] = [tree]

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    print("Decision path for DecisionTree {0}".format(j))
    node_indicator = tree.decision_path(X)
    leave_id = tree.apply(X)
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    features[j] = [tree.tree_.feature[0]]
    #tree_dict['tree'+str(j)] = ['feature': tree.tree_.feature, 


    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] != node_id:
            continue

        if (X[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X_train[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 X[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))

    print('\n\n')

df = pd.Series(data=features)
df = df.value_counts().rename_axis('feature').reset_index(name='TestStatistic')
df['pvalues'] = [st.binom_test(x, 20, 1/20, alternative='greater') for x in df.counts]
fdrs = correct.fdrcorrection(df.pvalues,  method='negcorr' )[1]
df['FDR']= fdrs

'''
This stack overflow page contains info for how to access 
individual trees and decision paths :
    
    https://stackoverflow.com/questions/48869343/decision-path-for-a-random-forest-classifier
    


'''

