# Experiment 5: Binary Classification on Breast Cancer using Decision Tree and Random Forest

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement and compare Decision Tree and Random Forest classifiers on the Wisconsin Diagnostic Breast Cancer dataset, evaluating hyperparameter tuning with 5-fold cross-validation, feature importance rankings, learning curves, and model stability.

## Dataset
- Name: Wisconsin Diagnostic Breast Cancer (WDBC)
- Instances: 569 samples (357 Benign, 212 Malignant)
- Features: 30 continuous attributes computed from digitized fine needle aspirate images
- Target: Binary diagnosis (0 = Benign, 1 = Malignant)

## Methodology
- Preprocessing: Feature standardization with StandardScaler and stratified 80/20 train-test split.
- Decision Tree: Grid search over splitting criteria (Gini, Entropy), maximum tree depth, minimum samples split, and minimum samples per leaf.
- Random Forest: Grid search over ensemble size (n_estimators), tree depth, feature subsampling (max_features='sqrt'), and split criteria.
- Visual Diagnostics: Learning curves, tree structure rendering, Gini feature importance plots, ROC curves, Precision-Recall curves, and Confusion Matrices.
- Performance Comparison: Evaluation across 5-fold cross-validation and held-out test set for accuracy, precision, recall, and F1-score.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-5.ipynb
```
