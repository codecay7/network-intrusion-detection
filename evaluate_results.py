import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

csv_path = Path("results/metrics/nids_detection_results.csv")
df = pd.read_csv(csv_path)

required = {"label", "prediction", "attack_probability"}
missing = required - set(df.columns)

if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")

y_true = df["label"].astype(int)
y_pred = df["prediction"].map({"NORMAL": 0, "ATTACK": 1})
y_prob = pd.to_numeric(df["attack_probability"], errors="raise")

if y_pred.isna().any():
    raise SystemExit("Unexpected values in prediction column.")

y_pred = y_pred.astype(int)
tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "dataset": "UNSW-NB15 official test set",
    "threshold": 0.70,
    "records": len(df),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "attack_precision": float(precision_score(y_true, y_pred)),
    "attack_recall": float(recall_score(y_true, y_pred)),
    "attack_f1": float(f1_score(y_true, y_pred)),
    "roc_auc": float(roc_auc_score(y_true, y_prob)),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "false_positive_rate": float(fp / (fp + tn)),
    "false_negative_rate": float(fn / (fn + tp)),
    "status": "Exploratory evaluation; test set previously used for threshold analysis",
}

out = Path("results/metrics/official_test_evaluation.json")
out.write_text(json.dumps(metrics, indent=2))

print(json.dumps(metrics, indent=2))
print(f"\nSaved report: {out}")
