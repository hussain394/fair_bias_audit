import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, equalized_odds_difference,
    selection_rate, false_positive_rate, false_negative_rate,
)
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "splits" / "test_with_predictions.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "mitigation"

st.set_page_config(page_title="Fairness & Bias Audit Dashboard", layout="wide")
st.title("Fairness & Bias Audit Dashboard")
st.caption("Audit model predictions for demographic parity and equalized odds across sensitive attributes.")

@st.cache_data
def load_test_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_mitigation_comparison(attr):
    file_path = REPORTS_DIR / f"comparison_{attr}.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

df = load_test_data()
if df is None:
    st.error(f"Data file not found at `{DATA_PATH}`. Please run data preparation and model training scripts first.")
    st.stop()


# --- Sidebar controls ---
st.sidebar.header("Audit Settings")
model_choice = st.sidebar.selectbox(
    "Model predictions",
    options=["pred_logistic_regression", "pred_xgboost"],
    format_func=lambda x: x.replace("pred_", "").replace("_", " ").title(),
)
sensitive_attr = st.sidebar.selectbox("Sensitive attribute", options=["sex", "race"])

y_true = df["y_true"]
y_pred = df[model_choice]
sf = df[sensitive_attr]

# --- Overall metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
col2.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")

dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
col3.metric("Demographic Parity Diff", f"{dp_diff:.3f}", help="0 = fair, closer to 1 = biased")
col4.metric("Equalized Odds Diff", f"{eo_diff:.3f}", help="0 = fair, closer to 1 = biased")

# --- Group-wise breakdown ---
st.subheader(f"Group-wise metrics by {sensitive_attr}")
mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    },
    y_true=y_true, y_pred=y_pred, sensitive_features=sf,
)
st.dataframe(mf.by_group.style.format("{:.3f}"))

fig, ax = plt.subplots(figsize=(6, 3.5))
mf.by_group["selection_rate"].plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Selection rate (predicted >50K)")
ax.set_title(f"Selection rate by {sensitive_attr}")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)

# --- Plain-English summary ---
st.subheader("Plain-English Summary")
groups = mf.by_group["selection_rate"]
max_group, min_group = groups.idxmax(), groups.idxmin()
gap_pct = (groups[max_group] - groups[min_group]) * 100
st.write(
    f"The model predicts a positive outcome (>50K income) for **{max_group}** "
    f"individuals **{gap_pct:.1f} percentage points** more often than for **{min_group}** "
    f"individuals, holding all else in the data as-is. "
    f"{'This is a substantial disparity.' if gap_pct > 10 else 'This is a moderate disparity.' if gap_pct > 3 else 'This disparity is relatively small.'}"
)

# --- Mitigation comparison, if available ---
st.subheader("Mitigation Comparison")
mit_df = load_mitigation_comparison(sensitive_attr)
if mit_df is not None:
    st.dataframe(mit_df.style.format({
        "accuracy": "{:.3f}", "f1": "{:.3f}",
        "demographic_parity_diff": "{:.3f}", "equalized_odds_diff": "{:.3f}",
    }))

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(mit_df["equalized_odds_diff"], mit_df["accuracy"])
    for _, row in mit_df.iterrows():
        ax2.annotate(row["method"], (row["equalized_odds_diff"], row["accuracy"]), fontsize=8, rotation=10)
    ax2.set_xlabel("Equalized Odds Difference (lower = fairer)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Fairness Trade-off")
    st.pyplot(fig2)
else:
    st.info(f"Run `mitigate_bias.py` with sensitive_attr='{sensitive_attr}' to see mitigation comparison here.")

if __name__ == "__main__":
    import subprocess
    print("Launching Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])