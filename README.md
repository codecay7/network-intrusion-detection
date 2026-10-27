# Machine Learning-Based Network Intrusion and Anomaly Detection System

An academic cybersecurity project investigating machine learning methods for
classifying network traffic as normal or potentially malicious.

## Objectives

- Prepare and inspect the UNSW-NB15 network intrusion dataset.
- Establish reproducible machine learning baselines.
- Evaluate models using precision, recall, F1-score, confusion matrices,
  and false-positive analysis.
- Compare model performance and computational cost.
- Develop a prototype dashboard to present evaluation results.

## Dataset

Primary candidate: UNSW-NB15.

Official source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Download the dataset separately. Raw and processed data are excluded from Git
to keep the repository lightweight.

## Workflow

1. Dataset inspection and data-quality checks.
2. Preprocessing and leakage prevention.
3. Baseline model training.
4. Evaluation on the designated test set.
5. Error analysis and model comparison.
6. Prototype dashboard and final report.

## Reproducibility

Experiments will record dataset versions, preprocessing choices, random seeds,
model settings, and evaluation metrics. No performance claims will be made
until experiments have been run.

## Project status

In development. Dataset inspection and model evaluation are pending.

## Technology

Python, pandas, NumPy, scikit-learn, Matplotlib, and Jupyter/Google Colab.
