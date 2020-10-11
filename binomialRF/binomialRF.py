# cython: profile=True

"""binomialRF: feature selection in random forests using random forests"""

import warnings
import numpy as np
import pandas as pd
import structure_dt as stdt
import graphs
import random
#from libc.math cimport log as clog
#from libc.math cimport exp
#cimport numpy as cnp
#cimport cython

# evaluate random forest ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
#from sklearn.ensemble import RandomForestRegressor as rf2
import sklearn.tree
import pandas as pd
import scipy.stats as st
import statsmodels.stats.multitest as correct

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.datasets import make_classification

class binomialRF:
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
    
    def __init__(self, X, y, ntrees, max_depth):
        self.X = X
        self.y = y
        self.ntrees = ntrees
        self.max_depth= max_depth
    
    def fit_model(self):
        model= rf(n_estimators=self.ntrees,
                  max_depth=self.max_depth)
        
        fitted_model = model.fit(self.X,self.y)
        
        return fitted_model
    
    
    def get_main_effects(self, rf_model):
        
        features= [[tree.tree_.feature[0]] for j,tree in enumerate(rf_model.estimators_)]
        df = pd.Series(data=features)

        ## calculate frequency of forot split vars
        main_effects_counts = df.value_counts().rename_axis('feature').reset_index(name='TestStatistic')
        return(main_effects_counts)
    
    def calculate_pvalue(self, main_effects_counts): 
        main_effects_counts['pvalues'] = [st.binom_test(x, 20, 1/20, alternative='greater') for x in main_effects_counts.TestStatistic]
        fdrs = correct.fdrcorrection(main_effects_counts.pvalues,  method='negcorr' )[1]
        main_effects_counts['FDR']= fdrs
        return(main_effects_counts)

