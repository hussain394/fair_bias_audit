# src/mitigate_bias.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, equalized_odds_difference,
    selection_rate, false_positive_rate, false_negative_rate,
)

def load_and_prepare(data_path="data/adult_income_raw.csv"):
    df = pd.read_csv(data_path).dropna()
    sensitive = df[["sex", "race"]].copy()
    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income"])

    cat_cols = X.select_dtypes(include="category").columns.tolist() + \
               X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y, sensitive

def fairness_summary(y_true, y_pred, sensitive_feature, label):
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "fpr": false_positive_rate,
            "fnr": false_negative_rate,
        },
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feature,
    )
    dp = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred)

    print(f"\n--- {label} ---")
    print(mf.by_group)
    print(f"Overall accuracy: {overall_acc:.4f} | F1: {overall_f1:.4f}")
    print(f"Demographic parity diff: {dp:.4f} | Equalized odds diff: {eo:.4f}")

    return {
        "method": label,
        "accuracy": overall_acc,
        "f1": overall_f1,
        "demographic_parity_diff": dp,
        "equalized_odds_diff": eo,
    }

def run_mitigation(sensitive_attr="sex"):
    X, y, sensitive = load_and_prepare()

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.25, random_state=42, stratify=y
    )
    sf_train = sens_train[sensitive_attr]
    sf_test = sens_test[sensitive_attr]

    results = []

    # --- Baseline (for comparison) ---
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_test)
    results.append(fairness_summary(y_test, preds, sf_test, "baseline"))

    # --- 1. Pre-processing: CorrelationRemover ---
    # CorrelationRemover needs the sensitive column INCLUDED in X, then removes
    # its linear correlation with everything else. We pass column name via index.
    X_train_cr = X_train.copy()
    X_test_cr = X_test.copy()
    sensitive_col_idx = list(X_train_cr.columns).index("sex") if sensitive_attr == "sex" else list(X_train_cr.columns).index("race")

    cr = CorrelationRemover(sensitive_feature_ids=[X_train_cr.columns[sensitive_col_idx]])
    X_train_cr_arr = cr.fit_transform(X_train_cr)
    X_test_cr_arr = cr.transform(X_test_cr)

    model_cr = LogisticRegression(max_iter=1000)
    model_cr.fit(X_train_cr_arr, y_train)
    preds_cr = model_cr.predict(X_test_cr_arr)
    results.append(fairness_summary(y_test, preds_cr, sf_test, "preprocessing_correlation_remover"))

    # --- 2. In-processing: ExponentiatedGradient ---
    base_est = LogisticRegression(max_iter=1000)
    eg = ExponentiatedGradient(estimator=base_est, constraints=DemographicParity())
    eg.fit(X_train, y_train, sensitive_features=sf_train)
    preds_eg = eg.predict(X_test)
    results.append(fairness_summary(y_test, preds_eg, sf_test, "inprocessing_exponentiated_gradient"))

    # --- 3. Post-processing: ThresholdOptimizer ---
    # Uses the ALREADY TRAINED baseline model, just adjusts thresholds per group
    to = ThresholdOptimizer(
        estimator=baseline,
        constraints="equalized_odds",
        predict_method="predict",
        prefit=True,
    )
    to.fit(X_train, y_train, sensitive_features=sf_train)
    preds_to = to.predict(X_test, sensitive_features=sf_test)
    results.append(fairness_summary(y_test, preds_to, sf_test, "postprocessing_threshold_optimizer"))

    # --- Save comparison table ---
    out_dir = "reports/mitigation"
    os.makedirs(out_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{out_dir}/comparison_{sensitive_attr}.csv", index=False)
    print(f"\n\n=== FULL COMPARISON TABLE ({sensitive_attr}) ===")
    print(df_results.to_string(index=False))
    print(f"\nSaved to {out_dir}/comparison_{sensitive_attr}.csv")

    return df_results

if __name__ == "__main__":
    run_mitigation(sensitive_attr="sex")