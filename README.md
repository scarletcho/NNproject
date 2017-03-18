# NNproject
- This is a **neural network implementation** in MATLAB.
- It is designed as a term project in Intro to Artificial Neural Networks class.
- This network is currently applied to a specific task of **predicting age by one's normalized fMRI data.**
- But it is applicable to other tasks as well.

## Key features
- **Task type**
  - classification
  - regression
- **Cost function**
  - MSE (Mean Square Error)
  - cross-entropy

## Cross-validation
- K-fold cross-validation
  - (nfolds) without inner CV for parameter optimization
- K-fold **nested** cross validation  
  - (nfolds) by (vfolds)  

## Not to overfit
- **L1** & **L2** regularization
  - and its corresponding **lambda**
  - or a candidate **list of lambdas** (if nested cv applied)
- **Dropout**
  - and **dropout rate**
- **Resampling** methods
  - none
  - Bayesian
  - GaussianNoise
