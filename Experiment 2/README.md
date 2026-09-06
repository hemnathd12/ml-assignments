# Experiment 2: Email Spam Classification using Naive Bayes and KNN

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement, tune, and compare probabilistic classifiers (Gaussian, Multinomial, and Bernoulli Naive Bayes) and instance-based classifiers (K-Nearest Neighbors) on the Spambase dataset for binary spam detection.

## Dataset
- Name: UCI Spambase Dataset
- Samples: 4,601 email records (1,813 Spam, 2,788 Ham)
- Features: 57 continuous attributes representing word and character frequencies and run-length statistics
- Target: Binary class (1 = Spam, 0 = Ham)

## Algorithms and Experiments
- Naive Bayes Variants: Implementation and evaluation of GaussianNB, MultinomialNB, and BernoulliNB.
- K-Nearest Neighbors: Performance analysis across varying values of k (1 to 25) with Euclidean, Manhattan, and Minkowski distance metrics.
- Algorithm Complexity and Efficiency: Benchmarking training and query times for Brute Force, KDTree, and BallTree neighbor search algorithms.
- Hyperparameter Optimization: Systematic tuning using GridSearchCV and RandomizedSearchCV.
- Cross-Validation: 5-fold stratified cross-validation for stability and variance assessment.
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC Curves, and Confusion Matrices.

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-2.ipynb
```
