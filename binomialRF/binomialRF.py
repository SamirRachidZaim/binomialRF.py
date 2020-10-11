# cython: profile=True

"""binomialRF: feature selection in random forests using random forests"""

import warnings
import pandas as pd
import numpy as np
import structure_dt as stdt
import graphs
import random
#from libc.math cimport log as clog
#from libc.math cimport exp
#cimport numpy as cnp
#cimport cython

# evaluate random forest ensemble for regression

import sklearn.tree
import scipy.stats as st
import statsmodels.stats.multitest as correct

from rpy2.robjects.packages import importr


from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import scipy.stats as st
import statsmodels.stats.multitest as correct



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
    
    X : data.frame or array
        The predictor matrix for which one wants to learn the features' 
        significance. 

    y : the label/outcome vector
        The y vector used to label the instances/observations in the
        design/predictor matrix, X. 
    
    
    ntrees : integer, greater than zero
        The (maximum) number of trees to build in fitting the model.  We
        highly recommend using early stopping rather than guesswork or
        grid search to set this parameter.
    
    subsample : float, default is .3
        The percentage of cases sampled by each decision tree to train each
        weak learner in the random forest classifier. 
    
    
    References:
    -----------
    
    Zaim, Samir Rachid, Colleen Kenost, ..., Hao Helen Zhang, and Yves A. 
    Lussier. "binomialRF: Interpretable combinatoric efficiency of random 
    forests to identify biomarker interactions." bioRxiv (2020): 681973.
    s
    """
    
    def __init__(self, X, y, ntrees, subsample):
        self.X = X
        self.y = y
        self.ntrees = ntrees
        self.subsample= subsample
    
    def fit_model(self):

        model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), max_samples=self.subsample, n_estimators=self.ntrees)
        fitted_model = model.fit(self.X,self.y)
        
        return fitted_model
    
    
    def get_main_effects(self, rf_model):
        
        features= [[tree.tree_.feature[0]] for j,tree in enumerate(rf_model.estimators_)]
        df = pd.Series(data=features)

        ## calculate frequency of forot split vars
        main_effects_counts = df.value_counts().rename_axis('feature').reset_index(name='TestStatistic')
        return(main_effects_counts)
    
    def calculate_naive_pvalue(self, main_effects_counts): 
        main_effects_counts['pvalues'] = [st.binom_test(x, 20, 1/20, alternative='greater') for x in main_effects_counts.TestStatistic]
        fdrs = correct.fdrcorrection(main_effects_counts.pvalues,  method='negcorr' )[1]
        main_effects_counts['FDR']= fdrs
        return(main_effects_counts)

    def calculate_cbinom(self):
        correlbinom = importr('correlbinom')

        ncols= len(self.X.columns)
        success_prob = 1 / ncols
        cbinom= correlbinom.correlbinom(self.subsample_percentage, success_prob, self.ntrees)
        return cbinom

    def calculate_correlated_pvalues(self, main_effects_counts, cbinom):
        main_effects_counts['correl_pvalue'] = [1-np.sum(cbinom[0:x]) for x in main_effects_counts.TestStatistic]
        fdrs = correct.fdrcorrection(main_effects_counts.correl_pvalue,  method='negcorr' )[1]
        main_effects_counts['FDR']= fdrs
        return main_effects_counts
  