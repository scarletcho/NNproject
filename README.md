# NNproject
- This is a **neural network implementation** in MATLAB.
- It is designed as a term project in Intro to Artificial Neural Networks class.
- This network is currently applied to a specific task of **predicting age by one's normalized fMRI data.**
- But it is applicable to other tasks as well.

## Key features
### 1. Define the model
- **Task type**
  - Classification
  - Regression
- **Cost function**
  - MSE (Mean Square Error)
  - Cross-entropy

</br>

### 2. Cross-validation
- K-fold cross-validation
  - (nfolds) without inner CV for parameter optimization
- K-fold **nested** cross validation  
  - (nfolds) by (vfolds)  

</br>

### 3. Not to overfit
- **L1** & **L2** regularization
  - and its corresponding **lambda**
  - or a candidate **list of lambdas** (if nested cv applied)
- **Dropout**
  - and **dropout rate**
- **Resampling** methods
  - None
  - Bayesian
  - GaussianNoise
