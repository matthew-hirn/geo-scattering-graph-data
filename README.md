# Geometric Scattering for Graph Data Analysis

This code computes geometric wavelet scattering transforms for graphical data based on the paper:

Feng Gao, Guy Wolf, Matthew Hirn<br/>
**"Geometric Scattering for Graph Data Analysis"**<br/>
[*Proceedings of the 36th International Conference on Machine Learning*](http://proceedings.mlr.press/v97/gao19e.html), PMLR 97:2122-2131, 2019


Geometric scattering coefficients could be used for various downstream tasks, and in the paper we considered four of them, namely graph classification of social network datasets, graph classification under low training data availability, dimention reduction and data exploration of ENZYME dataset. 

## Graph Classification of Social Network Datasets

There are six social network datasets commonly used for graph classification: COLLAB, IMDB-B, IMDB-M, REDDIT-B, REDDIT-M and REDDIT-12K. In addition, we also compare our results with 10 kernel and deep learning methods.

## Graph Classification Under Low Training Data Availability

Geometric scattering features are based on each graph without any training processes. In this section, we demonstrate the performance of the GS-SVM under low training-data availability and show that the scattering features can embed enough graph information that even
under extreme conditions (e.g. only 20% training data), they can still maintain relatively good classification results.

## Dimensionality reduction

In this section we that geometric scattering combined with PCA enables significant dimensionality reduction with only a small impact on classification accuracy.

## Data exploration: Enzyme class exchange preferences

ENZYME dataset is a biochem dataset that contains 600 graphs representing enzyme structures. In this section we show that scattering features are sufficiently rich to capture relations between enzyme classes.
