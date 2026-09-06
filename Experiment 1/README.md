# Experiment 1: Exploratory Data Analysis and Feature Evaluation

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To perform comprehensive exploratory data analysis across multiple tabular datasets and image data, implementing statistical functions for checking normality, identifying univariate and multivariate outliers, analyzing missing data, computing correlations, assessing feature importance, and running baseline classification.

## Datasets
- Iris: 150 samples, 4 continuous features, 3 classes
- Loan Amount: 30,000 records, 24 features
- Spambase: 4,601 records, 57 continuous features, binary classification
- Pima Indians Diabetes: 768 samples, 8 numerical predictors, binary classification
- MNIST Handwritten Digits: 70,000 image samples (28x28 grayscale pixels), 10 classes

## Methodological Workflow
- Missing Value Diagnostics: Identification of missing patterns and median/mode imputation.
- Distribution and Normality: Shapiro-Wilk and D'Agostino-Pearson tests alongside Q-Q plots.
- Outlier Identification: IQR boundaries, Z-score filters, Mahalanobis distance for multivariate anomalies, and Isolation Forest.
- Association and Multicollinearity: Pearson and Spearman correlation matrices with threshold-based pair extraction.
- Feature Relevance: Variance filtering, Mutual Information classification scoring, and Random Forest feature importance ranking.
- Image Domain EDA: Mean digit templates, pixel variance maps, PCA projection, and baseline Random Forest classification.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-1.ipynb
```
