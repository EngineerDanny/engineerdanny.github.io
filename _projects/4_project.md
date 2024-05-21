---
layout: page
title: Microbiome Network Analysis
description: This project involves the creation of a series of plots and graphs that represent the complex microbiome data from a scientific study.
img: assets/img/threshold_network.png
importance: 3
category: work
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/threshold_network_full.png" title="threshold_network_full" class="img-fluid rounded z-depth-1" width="700px" height="500px" %}
    </div>
</div>

## Overview
Microbiome are the microorganisms that live in a particular environment, such as the soil, gut, or water. 
These organisms form associations or interactions with each other, creating complex networks that can be challenging to analyze.
Some of the algorithms which are used to infer the network structure are Correlation-based (Pearson/Spearman), LASSO and Gaussian Graphical Models.
In this project, we were interested in analyzing the pairwise relationships between different entities in the microbiome data.
Using Pearson correlation as the main algorithm was a logical choice as it fits in this use case and it is widely used in the field of microbiome analysis.

This project involves the creation of a series of interactive plots and graphs (https://engineerdanny.github.io/EngineerDanny-necromass-figure-one-network/) that represent the complex microbiome data from a scientific study.
This analysis is done on **necromass** data from a study in University of Minnesota.
Necromass basically refers to the dead organic matter in the soil.
The data consists of samples from different conditions (AllSoilM1M3, HighMelanM1, HighMelanM3, LowMelanM1, LowMelanM3), and the visualizations are designed to identify correlations, distributions, and patterns within the dataset.
The visualizations provide a clear and concise way to present complex data, making it accessible to a broader audience.

## Plots Description
1. **Select Absolute Correlation Threshold**: This plot shows the relationship between different correlation thresholds and the number of edges, helping to determine the strength of the connections in the data.
It has the subtrain and validation error and the number of edges in the network against the absolute correlation threshold.
In ML, subtrain error is the error on the training data and validation error is the error on the validation data.
The number of edges in the network is the number of connections between the entities in the data.
To filter out the spurious correlations, we can set a threshold on the absolute value of the correlation coefficient.
This plot helps in selecting the optimal threshold for the network.
This corresponds to the number of edges where the validation error is minimum.


2. **Click Edge to Select Pair**: An interactive network diagram that allows users to explore the connections between different entities.
3. **Normalized Abundance for Selected Pair**: A scatter plot that displays the abundance of two variables over time, providing insights into their behavior and relationship.
4. **Select Sparsity or Observed Load**: This scatter plot visualizes the relationship between load and sparsity, which can be crucial for understanding the distribution of data points.
5. **Select Sparsity on Observed Load**: Similar to the third plot but with more data points, offering a detailed view of the sparsity in relation to the observed load.
6. **Select Sparsity per Load**: A line graph that represents various categories over time or another variable, highlighting the changes in sparsity with respect to the load.

## Methodology
The data was analyzed using statistical and computational techniques to extract meaningful patterns. The plots were generated using data visualization tools and libraries that support the rendering of scientific data.

## Importance
The visualizations provide a clear and concise way to present complex data, making it accessible to a broader audience. They are essential for:
- Communicating findings effectively in scientific research.
- Assisting in decision-making processes by revealing hidden trends.
- Facilitating peer review and collaborative efforts by providing a common visual language.

## Conclusion
The project showcases the power of data visualization in interpreting large datasets and emphasizes the need for effective communication in the field of data science.

