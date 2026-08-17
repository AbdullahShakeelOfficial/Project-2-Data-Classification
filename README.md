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