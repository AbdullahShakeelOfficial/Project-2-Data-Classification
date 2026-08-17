# Project 2 - Data Classification Using AI

A supervised machine learning classification project using the Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

## Project Overview

This project demonstrates the basic workflow of a supervised machine learning classification problem.

The Iris dataset is used to train and evaluate a KNN classification model. The project includes data loading, feature and target separation, train-test splitting, feature scaling, K selection, model training, prediction, and evaluation.

## Objectives

- Load and understand a dataset
- Separate features and target labels
- Split data into training and testing sets
- Scale numerical features
- Train a K-Nearest Neighbors classifier
- Test different K values
- Evaluate the model using multiple metrics
- Visualize KNN performance

## Dataset

The project uses the built-in Iris dataset provided by Scikit-learn.

### Dataset Details

- Samples: 150
- Features: 4
- Classes: 3

### Features

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

### Target Classes

- Setosa
- Versicolor
- Virginica

## Machine Learning Workflow

The project follows this workflow:

```text
Iris Dataset
     |
     v
Feature / Target Separation
     |
     v
Train-Test Split
     |
     v
Feature Scaling
     |
     v
KNN Model
     |
     v
K Value Testing
     |
     v
Predictions
     |
     v
Model Evaluation
## Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

This results in:

- 120 training samples
- 30 testing samples

A fixed random state is used to make the experiment reproducible.

## Feature Scaling

StandardScaler is used to standardize the numerical features.

The scaler is fitted only on the training data and then applied to both the training and testing data.

This helps prevent information from the test set from leaking into the training process.

## Algorithm

### K-Nearest Neighbors (KNN)

KNN classifies a new sample based on the classes of its nearest neighboring training samples.

Different K values from 1 to 15 were tested.

For this particular train-test split, the highest test accuracy was:

**96.67%**

Multiple K values achieved this accuracy. The implementation selects **K = 1** because it is the first K value reaching the maximum accuracy.

## Model Performance

| Metric | Result |
|---|---:|
| Best K selected | 1 |
| Highest Test Accuracy | 96.67% |
| Weighted F1 Score | 0.97 |
| Test Samples | 30 |

### Classification Results

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 0.91 | 1.00 | 0.95 |
| Virginica | 1.00 | 0.90 | 0.95 |

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

These metrics provide a more complete view of model performance than accuracy alone.

## K Value Accuracy Graph

The project tests K values from 1 to 15 and plots their corresponding accuracy using Matplotlib.

The graph helps visualize how changing the K value affects model performance.

## Technologies Used

- Python
- Scikit-learn
- Matplotlib

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt