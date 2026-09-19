# Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

Student: Hemnath D
Register Number: 3122247001022
Degree: M.Tech Integrated CSE
Course: ICS1512 - Machine Learning Algorithms Laboratory

## Objective
To implement a single-layer Perceptron Learning Algorithm (PLA) from scratch and a Multilayer Perceptron (MLP) trained with backpropagation on the English Handwritten Characters dataset, tune the MLP's activation function, optimizer, learning rate, batch size, architecture, loss function and regularization, and compare the two models with accuracy, precision, recall, F1-score, confusion matrices, micro/macro ROC curves and convergence curves.

## Dataset
- Name: English Handwritten Characters Dataset (Kaggle, dhruvildave/english-handwritten-characters-dataset)
- Instances: 3,410 scanned character images, 62 classes (0-9, A-Z, a-z), 55 images per class
- Features: 1,024 after preprocessing (grayscale, inverted, cropped to the ink bounding box, resized to 32 x 32, scaled to [0, 1])
- Download with `kagglehub.dataset_download("dhruvildave/english-handwritten-characters-dataset")` and place `english.csv` and the `Img` folder in this directory

## Methodology
- Split: stratified 80/20 train+validation/test, then 2,294 training, 434 validation and 682 test images.
- PLA: one-vs-rest perceptrons with step activation and the update w = w + eta (y - y_hat) x, implemented in NumPy; best-validation-epoch weights kept.
- MLP: PyTorch fully connected network trained with mini-batch backpropagation and early stopping; tuning in seven stages (activation, optimizer x learning rate, batch size, architecture, loss function, regularization, data augmentation), three seeds per configuration.
- Final MLP: one hidden layer of 256 ReLU units, SGD with momentum 0.9, learning rate 0.05, batch size 32, cross-entropy, weight decay 1e-4; checked with 5-fold cross-validation.
- Evaluation: accuracy, macro precision/recall/F1, micro and macro ROC-AUC, confusion matrices, per-class F1 and training error vs epoch curves.

## Results
- PLA: test accuracy 64.37%, macro F1 0.640, macro ROC-AUC 0.960
- MLP (tuned): test accuracy 73.17%, macro F1 0.731, macro ROC-AUC 0.985

## Installation and Execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook expt-9.ipynb
```
