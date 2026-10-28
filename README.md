# SENTINEL NIDS
## Machine Learning-Based Network Intrusion and Anomaly Detection System

SENTINEL NIDS is an academic machine-learning prototype that classifies
network-flow records as normal or potentially malicious using the UNSW-NB15
dataset and a Random Forest classifier.

## Features

- UNSW-NB15 dataset inspection and preprocessing
- Random Forest and Decision Tree baseline models
- Classification using 42 model input features
- Configurable decision threshold
- CSV upload and batch prediction through Streamlit
- Predicted traffic distribution and ground-truth comparison
- Confusion matrix, precision, recall, F1-score and accuracy
- CSV export of prediction results

## Dataset

Dataset: UNSW-NB15  
Official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Download the dataset separately. Raw datasets, processed data and trained
model binaries are excluded from Git. The model files and configuration must
be available locally before launching the dashboard.

## Technology

- Python
- pandas and NumPy
- scikit-learn and joblib
- Streamlit
- Matplotlib and Seaborn

## Run locally

Create and activate a virtual environment, then install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Ensure the following files are available:

- models/random_forest.joblib
- models/decision_tree.joblib
- models/model_config.json
- results/metrics/random_forest_threshold_070_validation.json

Launch the dashboard:

    streamlit run app.py

## Recorded results

### Validation at threshold 0.70

- Accuracy: 94.10%
- Attack precision: 98.77%
- Attack recall: 92.48%
- Attack F1-score: 95.52%
- False-positive rate: 2.46%
- ROC-AUC: 0.9938

### Official test-set exploration at threshold 0.70

- Accuracy: 92.04%
- Attack precision: 92.19%
- Attack recall: 93.45%
- Attack F1-score: 92.82%
- ROC-AUC: 0.9832

**Evaluation limitation:** The official test set was used during threshold
exploration. These test-set results are exploratory and should not be described
as an untouched, independent final evaluation.

## Limitations

This prototype classifies uploaded CSV flow records. It does not capture live
network packets, monitor a network interface, or confirm real security incidents.
Predictions should be treated as model outputs requiring further investigation.

## Project status

The baseline models and Streamlit dashboard have been implemented and tested.
Final documentation, repository review and independent evaluation remain part
of the project finalisation process.
