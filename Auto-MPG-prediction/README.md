# Predict Fuel Efficiency

This project aims to predict the fuel efficiency (miles per gallon) of vehicles using regression models. The dataset used is the Auto MPG dataset from the UCI Machine Learning Repository.

## Dataset
The dataset contains various attributes of vehicles, such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin. The 'mpg' column represents the target variable.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Feature Augmentation](#feature-augmentation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Comparison](#comparison)

## Introduction
The project aims to analyze the performance of different regression models on an automotive dataset. It includes functionalities to preprocess the data, augment features, train regression models, and evaluate their performance.

## Data Preprocessing
The data preprocessing phase involves handling missing values and converting relevant columns to the appropriate data types. Specifically, the `horsepower` column's missing values are filled using the K-Nearest Neighbors imputation. The `car name` column is dropped, as it's not relevant for modeling.

## Feature Augmentation
The project introduces augmented data to enhance model generalization. The `add_features` function computes additional features like `displacement_per_cylinder` and `power_to_weight_ratio`. The `augment_data` function generates synthetic instances by perturbing existing data with Gaussian noise. This process aims to increase model robustness by providing a broader range of data instances.

## Model Training
The project employs several regression models for training. The following regression models are used:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression
- Decision Tree Regression
- Gradient Boosting Regression
- Stacking Regressor

Hyperparameter tuning is applied to models such as Random Forest, Decision Tree, and Gradient Boosting using GridSearchCV.

## Model Evaluation
Models are trained and evaluated using both the original and augmented datasets. The training and testing scores of each model are computed. This helps to assess how well the models fit the data and how well they generalize to unseen instances.

## Comparison
The comparison between models trained on original and augmented data is presented in a DataFrame. The DataFrame provides a side-by-side view of the training and testing scores of each model. This enables an easy comparison of model performance under different conditions.

### Original Data

| Model              | Train Score | Test Score | MAE      |
|--------------------|-------------|------------|----------|
| Random Forest      | 0.983136    | 0.900052   | 1.782004 |
| Decision Tree      | 0.948539    | 0.860861   | 2.230840 |
| Gradient Boosting  | 0.974110    | 0.893638   | 1.908018 |
| Linear Regression  | 0.852843    | 0.837234   | 2.313014 |
| Ridge Regression   | 0.848751    | 0.852070   | 2.321405 |
| Lasso Regression   | 0.807917    | 0.827548   | 2.580218 |
| Stacking Regressor | 0.965137    | 0.896407   | 1.794147 |

### Augmented Data

| Model              | Train Score | Test Score | MAE      |
|--------------------|-------------|------------|----------|
| Random Forest      | 0.994433    | 0.921049   | 1.292312 |
| Decision Tree      | 1.000000    | 0.858823   | 1.071250 |
| Gradient Boosting  | 0.999912    | 0.937564   | 0.931371 |
| Linear Regression  | 0.856113    | 0.835599   | 2.454077 |
| Ridge Regression   | 0.854985    | 0.834194   | 2.458263 |
| Lasso Regression   | 0.817825    | 0.795067   | 2.792019 |
| Stacking Regressor | 0.998858    | 0.930702   | 1.045923 |
