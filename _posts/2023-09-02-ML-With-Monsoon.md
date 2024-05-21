---
toc: true
layout: post
description: This post explains some of the basic concepts you will need in your machine learning workflow using monsoon.
badges: true
categories: [python, git, github, ml, monsoon]
title: ML with Monsoon
featured: true
---


## Introduction
Welcome to the ML with Monsoon project! 
This blog post will guide you through some of the essential steps in setting up and executing machine learning workflows using the Monsoon HPC cluster. 
Monsoon is a high-performance computing (HPC) cluster at Northern Arizona University. 
It is a shared resource that provides researchers with the computational power needed to run large-scale simulations and data analysis.
Although Monsoon is a powerful tool, it can be challenging to use for beginners. 
This blog post aims to provide a step-by-step guide to help you get started with machine learning on Monsoon.
The concepts covered in this project will be applicable to any HPC cluster, so even if you are not using Monsoon, you can still benefit from this guide.
We will cover how to connect to Monsoon, perform interactive debugging, submit batch jobs, and visualize key results.

## Connecting to Monsoon
To start using Monsoon, you can connect via the dashboard or through a secure shell (ssh).

- Access through the monsoon [dashboard](https://ondemand.hpc.nau.edu/pun/sys/dashboard/)
Navigate to the Monsoon Dashboard and log in with your credentials.

- Access through the secure shell (ssh).
Open your terminal and run the following command, replacing <username> with your actual NAU username. 
You will be prompted to enter your password.
```bash
ssh -Y <username>@monsoon.hpc.nau.edu
```

## Interactive/Debug Work
For interactive or debugging tasks, you can request a compute node. 
Here are two examples:
- Request a node with 4GB of RAM and 1 CPU for 24 hours:
```bash
srun -t 24:00:00 --mem=4GB --cpus-per-task=1 --pty bash
```
- Request a node with 8GB of RAM and 1 CPU for 1 hour to run a Python script.
```bash
srun -t 1:00:00 --mem=8GB --cpus-per-task=1 python analysis.py
```

## Submitting Jobs
For longer-running tasks, you can write a job script and submit it to the queue. 
Here is an example of a job script (jobscript.sh):
```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/scratch/da2343/output.txt 
#SBATCH --error=/scratch/da2343/error.txt 
#SBATCH --time=20:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
python analysis.py
```

Submit the job script using the following command:
```bash
sbatch jobscript.sh
```

## Visualizing Results

### [Time Graph](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/job_arrays_intermediate)
Understanding the time performance of your machine learning models is crucial. 
The following graph illustrates the time taken for different stages in the workflow.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/time_graph.png" title="time_graph" class="img-fluid rounded z-depth-1" width="700px" height="500px" %}
    </div>
</div>


### [Algorithm Selection](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/job_arrays_advanced)
Choosing the right algorithm can significantly impact the performance and accuracy of your model. 
Below is a graph showing the accuracy of different parallel algorithms.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/parallel_algo_acc.png" title="parallel_algo_acc" class="img-fluid rounded z-depth-1" width="700px" height="500px" %}
    </div>
</div>

### [Hyper-Parameter Tuning](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/optimization)
Hyper-parameter tuning is essential for optimizing the performance of your machine learning models. 
The following graph shows the loss during hyper-parameter tuning.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/loss_df_01.png" title="loss_df_01" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


## Conclusion
In this post, we've walked you through the essential steps for setting up and managing machine learning workflows on the Monsoon HPC cluster. 
From connecting to Monsoon and performing interactive debugging, to submitting jobs and visualizing results, these guidelines are designed to help you efficiently utilize Monsoon's powerful resources.

Harnessing the capabilities of high-performance computing can significantly accelerate your machine learning projects, enabling you to handle larger datasets, run more complex models, and achieve faster results. 
By following the steps outlined here, you can optimize your workflow, reduce development time, and focus on what matters most—building and refining your machine learning models.