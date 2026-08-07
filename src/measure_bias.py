# src/measure_bias.py
import pandas as pd
import matplotlib.pyplot as plt
import os
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score, f1_score

def measure_bias(
    data_path="data/splits/test_with_predictions.csv",
    model_col="pred_xgboost",   # switch to "pred_logistic_regression" to compare
    out_dir="reports/fairness",
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    y_true = df["y_true"]
    y_pred = df[model_col]

    all_rows = []

    for attr in ["sex", "race"]:
        sensitive_feature = df[attr]

        # MetricFrame computes each metric PER GROUP automatically
        mf = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,   # % predicted positive (>50K)
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_feature,
        )

        print(f"\n=== Group-wise metrics by {attr} ({model_col}) ===")
        print(mf.by_group)

        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)

        print(f"Demographic parity difference: {dp_diff:.4f}  (0 = perfectly fair, closer to 1 = more biased)")
        print(f"Demographic parity ratio:      {dp_ratio:.4f}  (1 = perfectly fair, closer to 0 = more biased)")
        print(f"Equalized odds difference:     {eo_diff:.4f}  (0 = perfectly fair)")

        all_rows.append({
            "sensitive_attr": attr,
            "demographic_parity_difference": dp_diff,
            "demographic_parity_ratio": dp_ratio,
            "equalized_odds_difference": eo_diff,
        })

        # Save the group-wise table
        mf.by_group.to_csv(f"{out_dir}/{model_col}_by_{attr}.csv")

        # Chart: selection rate by group (this is the one people show in interviews)
        plt.figure(figsize=(7, 4))
        mf.by_group["selection_rate"].plot(kind="bar", color="steelblue")
        plt.title(f"Selection rate (predicted >50K) by {attr} — {model_col}")
        plt.ylabel("Selection rate")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{model_col}_selection_rate_by_{attr}.png")
        plt.close()

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(f"{out_dir}/{model_col}_summary.csv", index=False)

    print(f"\nOverall accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Overall F1: {f1_score(y_true, y_pred):.4f}")
    print(f"\nSaved fairness tables + charts to {out_dir}/")

    return summary_df

if __name__ == "__main__":
    # Run for both models so you have a full before-mitigation comparison
    for model_col in ["pred_logistic_regression", "pred_xgboost"]:
        measure_bias(model_col=model_col)