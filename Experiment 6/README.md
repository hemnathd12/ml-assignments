# Experiment 6: Binary Classification on Breast Cancer using Bagging, Boosting and Stacking

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare Bagging, Boosting (AdaBoost and Gradient Boosting), and Stacking ensemble methods on the Wisconsin Diagnostic Breast Cancer dataset, evaluating hyperparameter tuning with 5-fold cross-validation, feature importance, ROC analysis, and bias-variance trade-offs.

## Dataset
- Name: Wisconsin Diagnostic Breast Cancer (WDBC)
- Instances: 569 samples (357 Benign, 212 Malignant)
- Features: 30 numerical measurements describing cell nuclei
- Target: Binary class (0 = Benign, 1 = Malignant)

## Methodology
- Preprocessing: Binary label encoding, feature scaling using StandardScaler, and stratified 80/20 train-test split.
- Bagging: Grid search tuning tree estimators, sample subsampling fraction, and feature subsampling fraction.
- Boosting: Grid search tuning AdaBoost and Gradient Boosting across estimator counts, learning rates, and tree depths.
- Stacking: Heterogeneous ensemble combining Support Vector Classifier (SVM), Gaussian Naive Bayes, and Decision Tree base learners with a Logistic Regression meta-learner.
- Validation: 5-fold cross-validation fold-wise spread analysis and held-out test evaluation.
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-6.ipynb
```
