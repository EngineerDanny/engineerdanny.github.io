---
layout: page
title: ML with Monsoon
description: This project shows the code to explain some of the basic concepts you will need in your machine learning workflow using monsoon.
img: assets/img/12.jpg
importance: 1
category: work
related_publications: true
---


## Connecting to Monsoon
- Access through the monsoon [dashboard](https://ondemand.hpc.nau.edu/pun/sys/dashboard/)
- Access through the secure shell (ssh). On your terminal, run below and type your password:
```bash
ssh -Y <username>@monsoon.hpc.nau.edu
```


## Interactive/Debug Work
- Request a compute node with 4GB of RAM and 1 cpu for 24 hours
```bash
srun -t 24:00:00 --mem=4GB --cpus-per-task=1 --pty bash
```
- Request a compute node with 8GB of RAM and 1 cpu for 1 hour
```bash
srun -t 1:00:00 --mem=8GB --cpus-per-task=1 python analysis.py
```

## Submitting Jobs
You can also write your program and submit a job shell script for you to be placed in the queue.
An example job script (jobscript.sh) is:
```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/scratch/da2343/output.txt #SBATCH --error=/scratch/da2343/error.txt #SBATCH --time=20:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
python analysis.py
```
Submit the job script using:
```bash
sbatch jobscript.sh
```

## [Time Graph](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/job_arrays_intermediate)
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/time_graph.png" title="time_graph" class="img-fluid rounded z-depth-1" width="700px" height="500px" %}
    </div>
</div>


## [Algorithm Selection](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/job_arrays_advanced)
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/parallel_algo_acc.png" title="parallel_algo_acc" class="img-fluid rounded z-depth-1" width="700px" height="500px" %}
    </div>
</div>

## [Hyper-Parameter Tuning](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/optimization)
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/loss_df_01.png" title="loss_df_01" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
