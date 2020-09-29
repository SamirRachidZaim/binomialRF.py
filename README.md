# binomialRF.py
Python implementation of binomialRF

## Summary

The binomialRF.py is the python implementation of the original R
package "binomialRF" by Rachid Zaim (2020): 

The binomialRF package is a random forest-based feature selection
package that provides a feature selection algorithm to be used in
randomForest classifiers. Treating each tree as a quasi binomial
stochastic process in a random forest,
binomialRF determines a feature’s importance by how often they are
selected in practice vs. as expected by random chance. Given that
trees are co-dependent as they subsample the same data, a theoretical
adjustment is made using a generalization of the binomial distribution
that adds a parameter to model correlation/association between trials.

## References

- Zaim, Samir Rachid, Colleen Kenost, Joanne Berghout, Wesley Chiu, Liam Wilson, Hao Helen Zhang, and Yves A. Lussier. "binomialRF: Interpretable combinatoric efficiency of random forests to identify biomarker interactions." bioRxiv (2020): 681973.


