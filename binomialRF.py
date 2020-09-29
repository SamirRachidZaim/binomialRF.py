# cython: profile=True

"""binomialRF: feature selection in random forests using random forests"""

import warnings
import numpy as np
import pandas as pd
import structure_dt as stdt
import graphs
import random
from libc.math cimport log as clog
from libc.math cimport exp
cimport numpy as cnp
cimport cython

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.datasets import make_classification


class binomialRF(object):
    """binomialRF: a correlated binomial process to select features using
    random forests. Requires sci-kit learn's random forest implementation.
    
    binomialRF.py is a python implementation of Rachid Zaim's binomialRF:
        https://github.com/SamirRachidZaim/binomialRF,
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03718-9
        
    Feature selection uses a correlation adjustment to treat for data-dependence
    on trees, and is used to develop a p-value-based ranking for features in 
    a random forest classifier. 
    
    Parameters:
    -----------
    num_trees : integer, greater than zero
        The (maximum) number of trees to build in fitting the model.  We
        highly recommend using early stopping rather than guesswork or
        grid search to set this parameter.
    
    max_depth : int, default is 3
        The maximum depth to use when building trees. This is an important
        parameter.  Smaller values are more likely to underfit and 
        larger values more likely to overfit.
    
    feat_sample_by_tree : float, default is 1
        The percentage of features to consider for each tree.
        Works multiplicatively with feat_ample_by_node.
    
    
    
    References:
    -----------
    
    Zaim, Samir Rachid, Colleen Kenost, ..., Hao Helen Zhang, and Yves A. 
    Lussier. "binomialRF: Interpretable combinatoric efficiency of random 
    forests to identify biomarker interactions." bioRxiv (2020): 681973.
    
    """
    
    def __init__(self, X, y,  num_trees, max_depth, feat_sample_by_tree):
        self.X= X
        self.y= y
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.feat_sample_by_tree = feat_sample_by_tree
        
        if num_trees < 50:
            warnings.warn('Need more trees to train the random forest')
        if max_depth is <1:
            warnings.warn('Trees require a depth of at least a root node')
        if feat_sample_by_tree < 2:
            warnings.warn('Need more features to train the random forest')
       

        

    def fit():

        ''' pass parameters for random forest '''
        model = rf(n_estimators=self.num_trees, 
           max_depth=self.max_depth, 
           feat_sample_by_tree = self.feat_sample_by_tree)
        
    
    ''' fit random forest to X, y data '''     
        rfObject = model.fit(self.X,self.y)
    
    
    '''return rfObject '''
        return(rfObject)
        
    
        
        
        
        
        
        
        
        
        
        
        