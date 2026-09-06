# Experiment 4: Binary Classification on Spambase using Logistic Regression and SVM

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To perform binary email classification on the Spambase dataset using Logistic Regression and Support Vector Machines (SVM), evaluating L1/L2 regularization penalties, kernel transformations, and hyperparameter tuning with 5-fold cross-validation.

## Dataset
- Name: UCI Spambase Dataset
- Instances: 4,601 email records (1,813 Spam, 2,788 Ham)
- Features: 57 numerical predictors (frequencies of specific keywords, special characters, and capital run-length measures)
- Label: Binary (1 = Spam, 0 = Ham)

## Methodology
- Preprocessing: Standard feature scaling with StandardScaler and stratified 80/20 train-test splitting.
- Logistic Regression: Baseline modeling and hyperparameter optimization across regularization penalties (L1 Lasso, L2 Ridge), regularization strengths C, and solvers (liblinear, saga).
- Support Vector Machines: Training and evaluation across Linear, Radial Basis Function (RBF), Polynomial, and Sigmoid kernels, followed by grid search over penalty C, kernel coefficient gamma, and polynomial degree.
- Validation: 5-fold cross-validation for hyperparameter tuning and out-of-fold generalization assessment.
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC Curves, and Confusion Matrices.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-4.ipynb
```
