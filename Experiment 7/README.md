# Experiment 7: Dimensionality Reduction and Model Evaluation (With and Without PCA)

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To study the effect of dimensionality reduction using Principal Component Analysis (PCA) on the performance of ten machine learning classifiers. The assignment evaluates all models under both No-PCA and With-PCA conditions across four benchmark datasets from Experiments 1 to 6, performing hyperparameter tuning, 5-fold cross-validation, and statistical significance testing.

## Datasets
- Iris: 150 samples, 4 continuous features, 3 classes (reduced to 2 components, 95.81% variance explained)
- Breast Cancer Wisconsin Diagnostic (WDBC): 569 samples, 30 features, binary classification (reduced to 7 components, 91.01% variance explained)
- Spambase: 4,601 samples, 57 features, binary classification (reduced to 43 components, 90.39% variance explained)
- Optical Recognition of Handwritten Digits: 1,797 samples, 64 features, 10 classes (reduced to 31 components, 90.05% variance explained)

## Models Evaluated
1. Support Vector Machine (SVM)
2. Gaussian Naive Bayes
3. k-Nearest Neighbors (KNN)
4. Logistic Regression
5. Decision Tree (CART)
6. Random Forest
7. AdaBoost
8. Gradient Boosting
9. Histogram-Based Gradient Boosting (HistGradientBoosting)
10. Stacking (Base learners: SVM, KNN, Random Forest, Decision Tree; Meta-learner: Logistic Regression)

## Methodological Workflow
- Feature Preprocessing: Leakage-free standardization using StandardScaler fitted strictly on training folds.
- Dimensionality Reduction: Principal Component Analysis (PCA) fitted on training partitions, with the number of components selected automatically per dataset as the smallest k reaching 90% cumulative explained variance.
- Hyperparameter Tuning: Exhaustive grid search over defined parameter spaces evaluated with 5-fold cross-validation independently for No-PCA and With-PCA.
- Cross-Validation: Stratified 5-fold cross-validation recording fold-wise accuracy, F1-scores, and overall means.
- Statistical Significance Testing: Paired two-sample Student t-test and Wilcoxon Signed-Rank test directly comparing paired fold metrics between No-PCA and With-PCA conditions.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-7.ipynb
```
