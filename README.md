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

## Main Effects 

The main effects binomialRF model detects feature importance using a 1-sided correlated binomial test. Treating each tree as a stochastic but correlated Bernoulli process, the CDF of a correlated, but exchangeable binomial model is used to determine significance, and then adjusted for multiple comparisons. 

## Interactions
The k.binomialRF algorithm extends the main effects search to K-way (or multiway) interactions treating sequential splits in a decision path as an interaction of features. The same correlated but exchangeable model is used to determine significance, with the only difference being that the probability of success is normalized by (2^{k-1})^-1, as the sequence of splits for interactions can occur up to 2^(k-1) times in a tree of depth K. 

## References

- Zaim, Samir Rachid, Colleen Kenost, Joanne Berghout, Wesley Chiu, Liam Wilson, Hao Helen Zhang, and Yves A. Lussier. "binomialRF: Interpretable combinatoric efficiency of random forests to identify biomarker interactions." bioRxiv (2020): 681973.

## Manuscript

- https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03718-9