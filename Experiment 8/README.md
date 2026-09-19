# Experiment 8: Clustering Human Activity Recognition Data using K-Means, DBSCAN and Hierarchical Clustering

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To cluster the UCI Human Activity Recognition (HAR) smartphone dataset with K-Means, DBSCAN and Hierarchical Agglomerative Clustering, choose the number of K-Means clusters with the elbow and silhouette methods, tune DBSCAN's eps and min_samples, compare linkage criteria for hierarchical clustering, and evaluate all three algorithms with internal (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external (ARI, NMI) metrics against the true activity labels.

## Dataset
- Name: Human Activity Recognition Using Smartphones (UCI Machine Learning Repository)
- Instances: 10,299 windows (train and test partitions merged) from 30 volunteers
- Features: 561 time and frequency domain features per 2.56 s window, already scaled to [-1, 1]
- Activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING (used only for evaluation)
- Download the archive from https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones and extract the `UCI HAR Dataset` folder into this directory

## Methodology
- Preprocessing: StandardScaler followed by PCA to 10 components (70.2% variance), used as the common feature space for all three algorithms; PCA-2D and t-SNE used for visualization.
- K-Means: k = 2 to 8 with 10 restarts, elbow (WCSS) and silhouette curves, k = 6 selected; sensitivity to k, initialization and feature space checked.
- DBSCAN: k-distance plot to locate eps, grid over eps (2.5 to 7.0) and min_samples (5, 10, 20), selection by silhouette with noise below 10%.
- Hierarchical: Ward linkage with dendrogram on all 10,299 windows, compared with single, complete and average linkage at 6 clusters.
- Evaluation: Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, Hungarian-mapped confusion matrices and per-activity recall.

## Results
- K-Means (k = 6): ARI 0.427, NMI 0.562
- DBSCAN (eps = 6.0, min_samples = 20): 3 clusters, 8.1% noise, ARI 0.321, NMI 0.500 (best internal metrics)
- Hierarchical Ward (6 clusters): ARI 0.490, NMI 0.610 (best agreement with activities)

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-8.ipynb
```
