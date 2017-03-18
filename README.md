# NNproject
This is a neural network implementation in MATLAB submitted as a term project in Intro to Artificial Neural Networks class.
This network is currently applied to a specific task of predicting age by one's normalized fMRI data, but applicalbe to other tasks as well.

## Features
- choice of cost function
  - MSE
  - cross-entropy
- task type
  - classification
  - regression
- k-fold nested cross validation
  - (nfolds) by (vfolds)
- k-fold cross-validation
  - (nfolds) without inner CV for parameter optimization
- L1-L2 regularization
  - and a candidate list of their lambda values
- dropout
  - and dropoutRate
- resampling
  - 'none'
  - 'Bayesian'
  - 'GaussianNoise'
