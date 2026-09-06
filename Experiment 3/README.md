# Experiment 3: Loan Amount Prediction using Linear and Regularized Regression

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To predict loan sanction amounts using Ordinary Least Squares (OLS) Linear Regression, Ridge, Lasso, and Elastic Net regression models, performing hyperparameter optimization, coefficient shrinkage analysis, and automatic feature selection.

## Dataset
- Name: Kaggle Loan Sanction Dataset
- Records: 30,000 applicant rows, 24 columns
- Target Variable: Loan Sanction Amount (USD)
- Key Predictors: Loan Amount Request, Property Price, Current Loan Expenses, Credit Score, Income

## Methodology
- Data Cleaning: Removal of identifier columns, missing target rows, and invalid sentinel values (-999).
- Preprocessing: Median imputation for numeric features, category-code encoding for categoricals, standardized scaling via StandardScaler, and 80/20 train-test splitting.
- Baseline Model: Unregularized Ordinary Least Squares (OLS) Linear Regression.
- Hyperparameter Tuning: 5-fold cross-validation grid search to identify optimal regularization strength alpha for Ridge, Lasso, and Elastic Net, and mixing ratio l1_ratio for Elastic Net.
- Regularization and Shrinkage Analysis: Comparing L1 and L2 norm penalties across models and tracking training vs. validation RMSE curves.
- Feature Selection: Identification of features driven to exact zero by Lasso L1 penalty.
- Evaluation Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R2).

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-3.ipynb
```
