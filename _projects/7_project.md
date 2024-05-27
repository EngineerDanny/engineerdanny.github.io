---
layout: page
title: Advanced Machine Learning Techniques with PyTorch
description: Implementing advanced machine learning algorithms.
img: assets/img/p7_zip_nnet.png
importance: 4
category: work
---

## Introduction

In the rapidly evolving field of machine learning, staying abreast of the latest techniques and tools is crucial for developing effective and efficient models. 
This project showcases a comprehensive approach to implementing and evaluating advanced machine learning models using a combination of `PyTorch`, `Scikit-Learn`, and `Plotnine`. 
The primary goals of this project are to:

1. **Implement Custom Models**: Develop custom machine learning models, including a basic featureless model, and more sophisticated neural networks using `PyTorch`.
2. **Cross-Validation and Hyperparameter Tuning**: Utilize cross-validation techniques to optimize model performance and fine-tune hyperparameters for the best results.
3. **Data Preprocessing and Feature Scaling**: Prepare datasets with appropriate preprocessing steps, including feature scaling, to ensure model accuracy and robustness.
4. **Comparative Analysis**: Compare the performance of different models using a standardized evaluation framework.
5. **Visualization of Results**: Generate insightful visualizations using `Plotnine` to interpret the performance and behavior of models across different datasets and configurations.

The project is structured to cover various aspects of machine learning model development and evaluation. 
We start with data preparation, including loading and scaling datasets. 
We then define custom classes for models and datasets, enabling seamless integration with PyTorch's data handling capabilities. 
The core of the project involves training neural networks with cross-validation to identify the optimal number of epochs for each model.

Moreover, we implement a series of experiments to compare traditional machine learning models, like Logistic Regression and K-Nearest Neighbors, with our neural network models. 
The performance of these models is evaluated across multiple folds of cross-validation, ensuring a robust assessment.

Finally, the project leverages Plotnine for creating detailed diagnostic plots and visualizations. These plots help in understanding the training dynamics and the effectiveness of different models, making it easier to draw actionable insights.

By the end of this project, you'll have a solid understanding of how to build, tune, and evaluate advanced machine learning models using a combination of powerful Python libraries.



## Imports
To begin, we need to import several essential libraries for our machine learning tasks. 
These include pandas, numpy, plotnine for visualization, `PyTorch` for deep learning, and several modules from `scikit-learn` for data processing and modeling.

```python
import pandas as pd
import numpy as np
import plotnine as p9
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
```

## Custom Classes
### Featureless Model
The `FeaturelessModel` class is a simple model that predicts the majority class in the training data. It is used as a baseline model to compare the performance of more complex models.
```python
class Featureless:
    def fit(self, y):
        train_labels = y
        train_label_counts = pd.Series(train_labels).value_counts()
        self.featureless_pred_label = train_label_counts.idxmax()

    def predict(self, X):
        test_features = X
        test_nrow, test_ncol = test_features.shape
        return np.repeat(self.featureless_pred_label, test_nrow)
```

### CSV Dataset Class for PyTorch
This class helps in loading datasets in a format compatible with PyTorch.
```python
class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]

    def __len__(self):
        return len(self.labels)
```

### Torch Learner with Cross-Validation
This class helps in loading datasets in a format compatible with PyTorch.

```python
class TorchLearnerCV():
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        
    def fit(self, X, y):
        train_features = X
        train_labels = y
        np.random.seed(1)
        n_folds = 2
        fold_vec = np.random.randint(low=0, high=n_folds, size=train_labels.size)
        is_set_dict = {
            "subtrain": fold_vec != (n_folds-1),
            "validation": fold_vec == (n_folds-1),
        }
        set_features = {}
        set_labels = {}
        for set_name, is_set in is_set_dict.items():
            set_features[set_name] = train_features[is_set, :]
            set_labels[set_name] = train_labels[is_set]
        
        min_validation_loss = None        
        for max_epochs_index in range(1, self.max_epochs):
            learner = TorchLearner(
                max_epochs=max_epochs_index,
                batch_size=self.batch_size,
                step_size=self.step_size,
                units_per_layer=self.units_per_layer)
            learner.fit(set_features, set_labels)
            loss_df = learner.loss_df
            validation_loss_df = loss_df[loss_df["set_name"] == "validation"]
            validation_loss = validation_loss_df["loss"].min()
            if min_validation_loss is None or validation_loss <= min_validation_loss:
                min_validation_loss = validation_loss
                self.best_max_epochs = max_epochs_index
            print(f"epoch = {max_epochs_index}, min_val_loss = {min_validation_loss}")
             
        self.torch_learner = TorchLearner(
            max_epochs=self.best_max_epochs,
            batch_size=self.batch_size,
            step_size=self.step_size,
            units_per_layer=self.units_per_layer)
        self.torch_learner.fit(set_features, set_labels)
        self.loss_df = self.torch_learner.loss_df
        print(f"best_max_epochs: {self.best_max_epochs}")

    def decision_function(self, X):
        return self.torch_learner.decision_function(X)

    def predict(self, X):
        return self.torch_learner.predict(X)
```

### Torch Model Definition
Defines the structure of the neural network.

```python
class TorchModel(torch.nn.Module):
    def __init__(self, *units_per_layer):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        seq_args = []
        for layer_i in range(len(units_per_layer)-1):
            units_in = units_per_layer[layer_i]
            units_out = units_per_layer[layer_i+1]
            seq_args.append(torch.nn.Linear(units_in, units_out))
            if layer_i != len(units_per_layer)-2:
                seq_args.append(torch.nn.ReLU())     
        self.linear_relu_stack = torch.nn.Sequential(*seq_args)

    def forward(self, X):
        X = self.flatten(X)
        return self.linear_relu_stack(X)
```

### Torch Learner Class
Handles the training process for the neural network.

```python
class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.model = TorchModel(*self.units_per_layer)
        self.loss_fun = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.step_size)

    def take_step(self, X, y):
        loss_tensor = self.calc_loss_tensor(X, y)
        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()
        return loss_tensor.item()

    def calc_loss_tensor(self, X, y):
        X_features = X.float()
        y_labels = y.float()
        pred_tensor = self.model(X_features).reshape(len(y_labels))
        loss_tensor = self.loss_fun(pred_tensor, y_labels)
        return loss_tensor

    def fit(self, X, y):
        set_features = X
        set_labels = y        
        subtrain_csv = CSV(set_features["subtrain"], set_labels["subtrain"])
        subtrain_dl = torch.utils.data.DataLoader(subtrain_csv, batch_size=self.batch_size, shuffle=True)
        validation_csv = CSV(set_features["validation"], set_labels["validation"])
        validation_dl = torch.utils.data.DataLoader(validation_csv, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        loss_df_list = []
        for epoch in range(self.max_epochs):
            total_subtrain_loss = 0
            total_validation_loss = 0
            n_subtrain_batches = len(subtrain_dl)
            n_validation_batches = len(validation_dl)
            for batch_features, batch_labels in subtrain_dl:
                subtrain_loss_item = self.take_step(batch_features, batch_labels)
                total_subtrain_loss += subtrain_loss_item

            for batch_features, batch_labels in validation_dl:
                validation_loss_item = self.calc_loss_tensor(batch_features, batch_labels).item()
                total_validation_loss += validation_loss_item

            avg_subtrain_loss = total_subtrain_loss / n_subtrain_batches
            loss_df_list.append(pd.DataFrame({
                "set_name": "subtrain",
                "loss": avg_subtrain_loss,
                "epoch": epoch,
            }, index=[0]))
            
            avg_validation_loss = total_validation_loss / n_validation_batches
            loss_df_list.append(pd.DataFrame({
                "set_name": "validation",
                "loss": avg_validation_loss,
                "epoch": epoch,
            }, index=[0]))

        self.loss_df = pd.concat(loss_df_list)

    def decision_function(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.Tensor(X)).numpy().ravel()

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, 0)
```

## Data Preparation
Load and preprocess the spam and zip datasets.
    
```python
spam_df = pd.read_csv("./spam.data", header=None, sep=" ")
spam_features = spam_df.iloc[:, :-1].to_numpy()
spam_labels = spam_df.iloc[:, -1].to_numpy()

spam_mean = spam_features.mean(axis=0)
spam_sd = np.sqrt(spam_features.var(axis=0))
spam_scaled_features = (spam_features - spam_mean) / spam_sd

zip_df = pd.read_csv("./zip.test.gz", header=None, sep=" ")
is01 = zip_df[0].isin([0, 1])
zip01_df = zip_df.loc[is01, :]
zip_features = zip01_df.loc[:, 1:].to_numpy()
zip_labels = zip01_df[0].to_numpy()

data_dict = {
    "zip": (zip_features, zip_labels),
    "spam": (spam_scaled_features, spam_labels),
}
```

## Hyperparameter Training and Diagnostic Plot
Generate diagnostic plots for model performance.
    
```python
def hyperparameter_training_and_diagnostic_plot():
    for data_set, (input_mat, output_vec) in data_dict.items():
        ncol = input_mat.shape[1]
        sizes_dict = {
            "linear": (ncol, 1),
            "nnet": (ncol, 100, 10, 1),
        }
        for model_name, units_per_layer in sizes_dict.items():
            my_nn_cv = TorchLearnerCV(
                max_epochs=40,
                batch_size=5,
                step_size=0.01,
                units_per_layer=units_per_layer)
            my_nn_cv.fit(input_mat, output_vec)
            test_loss_df = my_nn_cv.loss_df
            gg = p9.ggplot(data=test_loss_df) +\
                p9.aes(x="epoch", y="loss", color="set_name") +\
                p9.geom_line() +\
                p9.scale_color_discrete(name="set_name") +\
                p9.labs(title=f"Loss for {data_set} dataset, {model_name} model")
            gg.save(f"./{data_set}_{model_name}.png")
```

<div class="row justify-content-sm-center">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/p7_spam_linear.png" title="p7_spam_linear" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/p7_spam_nnet.png" title="p7_spam_nnet" class="img-fluid rounded z-depth-1" %}
  </div>
</div>


## Experiments and Application
Run experiments and evaluate model performance.
```python
def experiments_and_application():
    test_acc_df_list = [] 
    for data_set, (input_mat, output_vec) in data_dict.items():
        k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
        for fold_id, indices in enumerate(k_fold.split(input_mat)):
            index_dict = dict(zip(["train", "test"], indices))
            param_dicts = [{'n_neighbors': [x]} for x in range(1, 21)]

            set_data_dict = {}
            for set_name, index_vec in index_dict.items():
                set_data_dict[set_name] = {
                    "X": input_mat[index_vec],
                    "y": output_vec[index_vec]
                }
            
            sizes_dict = {
                "linear": (input_mat.shape[1], 1),
                "deep": (input_mat.shape[1], 100, 10, 1)
            }
            predict_dict = {}
            for model_name, units_per_layer in sizes_dict.items():
                my_nn_cv = TorchLearnerCV(
                    max_epochs=40,
                    batch_size=5,
                    step_size=0.01,
                    units_per_layer=units_per_layer)
                my_nn_cv.fit(**set_data_dict["train"])
                predict_dict[model_name] = my_nn_cv.predict(set_data_dict["test"]['X'])

            pipe = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000))
            pipe.fit(**set_data_dict["train"])

            grid_search_cv = make_pipeline(StandardScaler(), 
            GridSearchCV(estimator=KNeighborsClassifier(),
                          param_grid=param_dicts, cv=5))
            grid_search_cv.fit(**set_data_dict["train"])

            featureless = Featureless()
            featureless.fit(set_data_dict["train"]['y'])

            test_data_x = set_data_dict["test"]['X']
            test_data_y = set_data_dict["test"]['y']

            pred_dict = {
                "LogisticRegressionCV": pipe.predict(test_data_x),
                "featureless": featureless.predict(test_data_x),
                "GridSearchCV+KNNC": grid_search_cv.predict(test_data_x),
                "TorchLearnerCV+Linear": predict_dict["linear"],
                "TorchLearnerCV+Deep": predict_dict["deep"]
            }
            for algorithm, pred_vec in pred_dict.items():
                test_acc_dict = {
                    "test_accuracy_percent": (pred_vec == test_data_y).mean() * 100,
                    "data_set": data_set,
                    "fold_id": fold_id,
                    "algorithm": algorithm
                }
                test_acc_df_list.append(pd.DataFrame(test_acc_dict, index=[0]))

    test_acc_df = pd.concat(test_acc_df_list)
    print(test_acc_df)

    gg = p9.ggplot() +\
        p9.geom_point(
            p9.aes(
                x="test_accuracy_percent",
                y="algorithm"
            ),
            data=test_acc_df) +\
        p9.facet_wrap("data_set")
    gg.save("./accuracy_facetted.png")
```

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/p7_accuracy_facetted.png" title="threshold_network_full" class="img-fluid rounded z-depth-1" width="600px" height="600px" %}
    </div>
</div> 

# Interpretation of Test Accuracy Results

The graph above shows the test accuracy percentages for different algorithms applied to two datasets: `spam` and `zip`. 
Each point represents the accuracy of a particular algorithm for a specific fold of cross-validation. 
The algorithms compared include a featureless model, linear and deep neural networks trained with cross-validation, logistic regression with cross-validation, and k-nearest neighbors with hyperparameter tuning using grid search.

## Key Observations

### Spam Dataset
- **Featureless Model**: The accuracy is consistently around 60%, indicating that the most frequent class constitutes 60% of the dataset. This serves as a baseline.
- **TorchLearnerCV+Linear**: The accuracy is significantly higher, around 90%, demonstrating that a simple neural network with a linear configuration performs well on the spam dataset.
- **TorchLearnerCV+Deep**: Similar to the linear model, the deep neural network achieves around 90% accuracy, suggesting that additional complexity does not significantly improve performance for this dataset.
- **LogisticRegressionCV**: Achieves close to 90% accuracy, indicating that logistic regression is quite effective for the spam dataset.
- **GridSearchCV+KNNC**: Also achieves close to 90% accuracy, showing that k-nearest neighbors with optimal hyperparameters can perform well on this dataset.

### Zip Dataset
- **Featureless Model**: The accuracy is around 65%, indicating the baseline performance for the `zip` dataset.
- **TorchLearnerCV+Linear**: The accuracy is around 85%, showing that a simple linear neural network configuration is effective but leaves room for improvement.
- **TorchLearnerCV+Deep**: Achieves nearly 100% accuracy, indicating that a deep neural network is highly effective for the `zip` dataset, likely capturing complex patterns in the data.
- **LogisticRegressionCV**: Accuracy is around 90%, showing that logistic regression is also quite effective for the `zip` dataset but slightly less so than the deep neural network.
- **GridSearchCV+KNNC**: The accuracy is around 95%, indicating that k-nearest neighbors with optimal hyperparameters perform very well on this dataset.

## Conclusion
- For the **spam dataset**, both simple and complex models (linear and deep neural networks, logistic regression, and k-nearest neighbors) perform similarly well, with accuracies around 90%.
- For the **zip dataset**, the deep neural network outperforms all other models, achieving near-perfect accuracy. 
Other models, including logistic regression and k-nearest neighbors, also perform well but slightly less so.

This analysis highlights the importance of model selection and hyperparameter tuning tailored to the specific characteristics of each dataset. 
Complex models like deep neural networks can capture more intricate patterns, leading to better performance in datasets with higher complexity.





