import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"
METRICS_PATH = (
    BASE_DIR / "results" / "metrics"
    / "random_forest_threshold_070_validation.json"
)

st.set_page_config(
    page_title="NIDS | Network Threat Monitor",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #0b1018;
    color: #e5edf5;
}
[data-testid="stSidebar"] {
    background: #101824;
    border-right: 1px solid #263345;
}
[data-testid="stMetric"] {
    background: #111c29;
    border: 1px solid #263548;
    padding: 18px;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #edf5ff;
}
div[data-testid="stFileUploader"] {
    background: #111c29;
    padding: 12px;
    border: 1px dashed #40536a;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def read_json(path):
    return json.loads(path.read_text())


st.sidebar.markdown("## 🛡️ SENTINEL NIDS")
st.sidebar.caption("NETWORK INTRUSION DETECTION")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Analyze Traffic", "Model Information"],
)

if not MODEL_PATH.is_file() or not CONFIG_PATH.is_file():
    st.error("Model or configuration file is missing.")
    st.stop()

config = read_json(CONFIG_PATH)
model = load_model()
threshold = float(config["decision_threshold"])

st.title("Network Threat Monitor")
st.caption("Machine Learning-Based Network Intrusion and Anomaly Detection")

if page == "Overview":
    st.markdown("### System overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model status", "Loaded")
    c2.metric("Decision threshold", f"{threshold:.2f}")
    c3.metric("Model type", "Random Forest")

    st.markdown("### Validation performance")

    if METRICS_PATH.is_file():
        metrics = read_json(METRICS_PATH)
        st.caption(
            "Offline validation results — not live network telemetry "
            "or official test-set performance."
        )

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        b.metric("Attack precision", f"{metrics['attack_precision']:.2%}")
        c.metric("Attack recall", f"{metrics['attack_recall']:.2%}")
        d.metric("Attack F1", f"{metrics['attack_f1']:.2%}")

        e, f = st.columns(2)
        e.metric("False-positive rate", f"{metrics['false_positive_rate']:.2%}")
        f.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    else:
        st.warning("Validation metrics file not found.")

    st.info(
        "Upload a CSV containing the 42 UNSW-NB15 model input features "
        "under Analyze Traffic to classify network-flow records."
    )

elif page == "Analyze Traffic":
    st.markdown("### Analyze network-flow records")
    st.write(
        "Upload a CSV containing the same 42 input features used during "
        "training. The `id`, `attack_cat`, and `label` columns must not "
        "be included as model inputs."
    )

    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, low_memory=False)

            required_features = list(model.feature_names_in_)
            missing = sorted(set(required_features) - set(data.columns))
            extra = sorted(set(data.columns) - set(required_features))

            if missing:
                st.error(
                    f"CSV is missing {len(missing)} required features: "
                    + ", ".join(missing[:15])
                )
                st.stop()

            if extra:
                st.warning(
                    "Extra columns will be ignored: "
                    + ", ".join(extra[:15])
                )

            X = data[required_features]

            if X.empty:
                st.warning("The uploaded CSV contains no records.")
                st.stop()

            if X.isnull().any().any():
                st.error(
                    "Missing values detected. Clean the data before "
                    "prediction; this model was trained without missing values."
                )
                st.stop()

            if st.button("Analyze traffic", type="primary"):
                with st.spinner("Analyzing network flows..."):
                    probabilities = model.predict_proba(X)[:, 1]
                    predictions = (probabilities >= threshold).astype(int)

                results = data.copy()
                results["attack_probability"] = probabilities
                results["prediction"] = [
                    "ATTACK" if p else "NORMAL" for p in predictions
                ]

                total = len(results)
                attacks = int(predictions.sum())
                normal = total - attacks

                st.markdown("### Detection summary")
                a, b, c = st.columns(3)
                a.metric("Flows analyzed", f"{total:,}")
                b.metric("Predicted attacks", f"{attacks:,}")
                c.metric("Predicted normal", f"{normal:,}")

                st.caption(
                    "Predictions are model outputs, not confirmed incidents. "
                    "Attack probability is the classifier's estimated score."
                )

                st.markdown("### Predicted traffic distribution")
                distribution = pd.DataFrame(
                    {"Flows": [normal, attacks]},
                    index=["Predicted normal", "Predicted attack"],
                )
                st.bar_chart(distribution, y="Flows")

                # Ground-truth comparison is possible only for labelled data.
                if "label" in data.columns:
                    actual = pd.to_numeric(data["label"], errors="coerce")
                    if actual.notna().all() and actual.isin([0, 1]).all():
                        actual = actual.astype(int)
                        predicted = pd.Series(predictions, index=data.index)

                        matrix = pd.crosstab(
                            actual.map({0: "Normal", 1: "Attack"}),
                            predicted.map({0: "Normal", 1: "Attack"}),
                            rownames=["Actual class"],
                            colnames=["Predicted class"],
                        ).reindex(
                            index=["Normal", "Attack"],
                            columns=["Normal", "Attack"],
                            fill_value=0,
                        )

                        st.markdown("### Ground-truth comparison")
                        st.caption(
                            "Evaluation on this uploaded labelled dataset. "
                            "Predictions are not confirmed security incidents."
                        )
                        st.dataframe(
                            matrix.style.format("{:,.0f}"),
                            use_container_width=True,
                        )

                        tn = int(matrix.loc["Normal", "Normal"])
                        fp = int(matrix.loc["Normal", "Attack"])
                        fn = int(matrix.loc["Attack", "Normal"])
                        tp = int(matrix.loc["Attack", "Attack"])
                        total = tn + fp + fn + tp

                        if total:
                            accuracy = (tn + tp) / total
                            precision = tp / (tp + fp) if tp + fp else 0
                            recall = tp / (tp + fn) if tp + fn else 0
                            f1 = (
                                2 * precision * recall / (precision + recall)
                                if precision + recall else 0
                            )

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Labelled accuracy", f"{accuracy:.2%}")
                            m2.metric("Attack precision", f"{precision:.2%}")
                            m3.metric("Attack recall", f"{recall:.2%}")
                            m4.metric("Attack F1", f"{f1:.2%}")
                    else:
                        st.info(
                            "The label column is not valid binary ground truth; "
                            "confusion matrix was not calculated."
                        )
                else:
                    st.info(
                        "No ground-truth label column was provided. "
                        "Only prediction counts are available; model accuracy "
                        "cannot be calculated for this upload."
                    )

                st.table(results.head(20))
                st.caption("Showing first 20 rows. Download the CSV for all results.")

                st.download_button(
                    "Download complete results CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="nids_detection_results.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Could not analyze this file: {exc}")

else:
    st.markdown("### Model information")
    st.write(f"**Dataset:** {config.get('dataset', 'Not specified')}")
    st.write(f"**Selected model:** {config.get('selected_model', 'Not specified')}")
    st.write(f"**Decision threshold:** {threshold:.2f}")
    st.write(f"**Target column:** {config.get('target_column', 'Not specified')}")
    st.write(f"**Positive class:** {config.get('positive_class', 1)}")
    st.write("**Input feature count:** 42")
    st.write(
        "**Threshold status:** "
        + config.get("threshold_status", "Not specified")
    )
    st.warning(
        "This prototype analyzes uploaded flow-feature CSVs. "
        "It does not yet capture live packets or monitor a network interface."
    )
