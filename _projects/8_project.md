---
layout: page
title: Regularized Neural Networks
description: Implementing a regularized neural network from scratch.
img: assets/img/p7_zip_nnet.png
importance: 4
category: work
---

## Introduction

Building a machine learning pipeline from scratch can be a daunting task, especially for those new to the field. This blog post aims to simplify this process by providing a step-by-step guide to creating a comprehensive machine learning pipeline. We'll cover everything from data preparation and model building to training, cross-validation, hyperparameter tuning, and visualization of results.

In this post, we'll walk through the Python code necessary to build and evaluate machine learning models using libraries such as pandas, numpy, scikit-learn, PyTorch, and plotnine. We will start by importing the essential libraries and creating custom classes for modeling. These classes include a featureless baseline model, a neural network implemented in PyTorch, and a custom cross-validation class. We will then move on to data preparation, where we'll preprocess datasets for training and testing.

Following data preparation, we'll train our models using various hyperparameters and generate diagnostic plots to visualize the results. Finally, we will apply our models to different datasets, comparing the performance of multiple algorithms to determine the best approach.

By the end of this post, you should have a clear understanding of how to build a robust machine learning pipeline, ready to be adapted and expanded for your specific needs. Let's dive in!

## Imports
The first step in any data science project is to import the necessary libraries. These libraries provide the tools for data manipulation, machine learning, and visualization.

```python
import pandas as pd
import numpy as np
import torch
import plotnine as p9
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
```

## Custom Classes
From the previous posts, we have created the `FeaturelessModel` class is a simple model that predicts the majority class in the training data. 

The `TorchModel` class defines a neural network using PyTorch. It allows for flexible architecture by specifying the number of hidden layers and units per layer.
```python
class TorchModel(torch.nn.Module):
    def __init__(self, n_hidden_layers, units_in_first_layer, units_per_hidden_layer=100):
        super(TorchModel, self).__init__()
        units_per_layer = [units_in_first_layer]
        for layer_i in range(n_hidden_layers):
            units_per_layer.append(units_per_hidden_layer)
        units_per_layer.append(1)
        seq_args = []
        for layer_i in range(len(units_per_layer)-1)):
            units_in = units_per_layer[layer_i]
            units_out = units_per_layer[layer_i+1]
            seq_args.append(torch.nn.Linear(units_in, units_out))
            if layer_i != len(units_per_layer)-2:
                seq_args.append(torch.nn.ReLU())
        self.stack = torch.nn.Sequential(*seq_args)

    def forward(self, feature_mat):
        return self.stack(feature_mat)
``` 

The `NumpyData` class converts numpy arrays into a PyTorch dataset, making it easier to work with DataLoader for batch processing.
```python
class NumpyData(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]

    def __len__(self):
        return len(self.labels)
```

The `MyCV` class handles cross-validation, iterating over parameter grids and tracking the best performing parameters.

```python
class MyCV:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv

    def fit(self, X, y):
        self.train_features = X
        self.train_labels = y
        self.best_params_ = {}
        np.random.seed(1)
        fold_vec = np.random.randint(low=0, high=self.cv, size=self.train_labels.size)
        best_mean_accuracy = 0
        for param_dict in self.param_grid:
            for param_name, [param_value] in param_dict.items():
                setattr(self.estimator, param_name, param_value)
                accuracy_list = []
                loss_df_list = []
                for test_fold in range(self.cv):
                    is_set_dict = {"validation": fold_vec == test_fold, "subtrain": fold_vec != test_fold}
                    set_features = {set_name: self.train_features[is_set, :] for set_name, is_set in is_set_dict.items()}
                    set_labels = {set_name: self.train_labels[is_set] for set_name, is_set in is_set_dict.items()}
                    self.estimator.fit(X=set_features, y=set_labels)
                    predicted_labels = self.estimator.predict(X=set_features["validation"])
                    accuracy = np.mean(predicted_labels == set_labels["validation"])
                    loss_df_list.append(self.estimator.loss_df)
                    accuracy_list.append(accuracy)
                mean_accuracy = np.mean(accuracy_list)
                if mean_accuracy > best_mean_accuracy:
                    best_mean_accuracy = mean_accuracy
                    self.best_params_[param_name] = param_value
                    setattr(self.estimator, param_name, self.best_params_[param_name])

        self.loss_mean_df = pd.concat(loss_df_list)
        print(self.loss_mean_df)

    def predict(self, X):
        return self.estimator.predict(X)
```