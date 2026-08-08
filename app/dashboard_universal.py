# app/dashboard_universal.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, equalized_odds_difference,
    selection_rate, false_positive_rate, false_negative_rate,
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity


st.set_page_config(page_title="Universal Fairness Audit Tool", layout="wide")
st.title("Universal Fairness & Bias Audit Tool")
st.caption("Upload any dataset → select target + sensitive feature → detect bias")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"Loaded **{df.shape[0]} rows x {df.shape[1]} columns**")
    st.dataframe(df.head())

    st.sidebar.header("Configure Audit")
    target_col = st.sidebar.selectbox("Target column", df.columns)
    sensitive_col = st.sidebar.selectbox(
        "Sensitive attribute", [c for c in df.columns if c != target_col]
    )
    target_values = df[target_col].dropna().unique().tolist()
    positive_class = st.sidebar.selectbox("Positive outcome", target_values)

    model_choice = st.sidebar.selectbox(
        "Baseline model",
        ["Logistic Regression", "XGBoost"],
        help="Logistic Regression is faster and more interpretable. XGBoost usually scores higher accuracy but can be less stable on very small or messy uploaded datasets.",
    )
    run_mitigation = st.sidebar.checkbox("Run Bias Mitigation", value=False)

    if st.sidebar.button("Run Audit"):
        try:
            # --- Clean + define target/sensitive/features ---
            work_df = df.dropna(subset=[target_col, sensitive_col]).copy()
            y = (work_df[target_col] == positive_class).astype(int)
            sensitive_feature = work_df[sensitive_col]
            X = work_df.drop(columns=[target_col, sensitive_col])

            # FIX 1: numeric = anything pandas calls "number" (covers int32/64,
            # float32/64, etc). categorical = literally everything else
            # (object, category, bool). No column silently falls through the cracks.
            numeric_cols = X.select_dtypes(include="number").columns.tolist()
            categorical_cols = X.columns.difference(numeric_cols).tolist()

            if len(numeric_cols) + len(categorical_cols) != X.shape[1]:
                st.warning("Some columns were not classified as numeric or categorical — check your data types.")

            # FIX 4: sparse_output=False keeps everything as a plain dense
            # array, which avoids inconsistent behavior between sklearn's
            # LogisticRegression and fairlearn's ExponentiatedGradient.
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ],
                remainder="drop",
            )

            # XGBoost needs the target as plain ints and handles NaNs natively;
            # LogisticRegression needs fully numeric, non-null input from the
            # ColumnTransformer above — both are satisfied by this pipeline.
            classifier = (
                LogisticRegression(max_iter=1000)
                if model_choice == "Logistic Regression"
                else XGBClassifier(eval_metric="logloss", random_state=42)
            )
            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ])

            # FIX 3: reset_index avoids any subtle index-alignment issues
            # between X_test / y_test / sf_test / preds later on.
            X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
                X, y, sensitive_feature,
                test_size=0.25, random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
            sf_train = sf_train.reset_index(drop=True)
            sf_test = sf_test.reset_index(drop=True)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.subheader("Baseline Model Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
            c2.metric("F1 Score", f"{f1_score(y_test, preds):.3f}")
            dp = demographic_parity_difference(y_test, preds, sensitive_features=sf_test)
            eo = equalized_odds_difference(y_test, preds, sensitive_features=sf_test)
            c3.metric("Demographic Parity Diff", f"{dp:.3f}")
            c4.metric("Equalized Odds Diff", f"{eo:.3f}")

            mf = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "selection_rate": selection_rate,
                    "false_positive_rate": false_positive_rate,
                    "false_negative_rate": false_negative_rate,
                },
                y_true=y_test, y_pred=preds, sensitive_features=sf_test,
            )
            st.subheader(f"Group Metrics by {sensitive_col}")
            st.dataframe(mf.by_group.style.format("{:.3f}"))

            # Chart — now guaranteed to run since everything above is inside
            # the try block; if anything failed, we'd see the error instead
            # of a silently missing chart.
            fig, ax = plt.subplots(figsize=(6, 3.5))
            mf.by_group["selection_rate"].plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(f"Selection Rate by {sensitive_col}")
            ax.set_ylabel("Selection rate")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            groups = mf.by_group["selection_rate"]
            max_g, min_g = groups.idxmax(), groups.idxmin()
            gap = (groups[max_g] - groups[min_g]) * 100
            st.info(f"Model predicts the positive outcome for **{max_g}** "
                    f"{gap:.1f} points more often than for **{min_g}**.")

            if run_mitigation:
                st.subheader("Bias Mitigation (Exponentiated Gradient)")
                st.caption("Mitigation uses Logistic Regression as its base estimator — Fairlearn's reductions API requires an estimator with `sample_weight` support, which XGBoost's sklearn wrapper doesn't reliably provide across versions.")
                with st.spinner("Training fairness-constrained model..."):
                    # FIX 2: transform ONCE using the already-fitted
                    # preprocessor from the pipeline — no redundant re-fit.
                    fitted_preprocessor = model.named_steps["preprocessor"]
                    X_train_t = fitted_preprocessor.transform(X_train)
                    X_test_t = fitted_preprocessor.transform(X_test)

                    mitigator = ExponentiatedGradient(
                        estimator=LogisticRegression(max_iter=1000),
                        constraints=DemographicParity(),
                    )
                    mitigator.fit(X_train_t, y_train, sensitive_features=sf_train)
                    mit_preds = mitigator.predict(X_test_t)

                mit_dp = demographic_parity_difference(y_test, mit_preds, sensitive_features=sf_test)
                mit_eo = equalized_odds_difference(y_test, mit_preds, sensitive_features=sf_test)

                comp = pd.DataFrame([
                    {"method": "baseline", "accuracy": accuracy_score(y_test, preds),
                     "f1": f1_score(y_test, preds), "dp_diff": dp, "eo_diff": eo},
                    {"method": "mitigated", "accuracy": accuracy_score(y_test, mit_preds),
                     "f1": f1_score(y_test, mit_preds), "dp_diff": mit_dp, "eo_diff": mit_eo},
                ])
                st.dataframe(comp.style.format({
                    "accuracy": "{:.3f}", "f1": "{:.3f}",
                    "dp_diff": "{:.3f}", "eo_diff": "{:.3f}",
                }))

        except Exception as e:
            # FIX 3: surface the real error instead of leaving you guessing
            # why the chart never showed up.
            st.error("Something went wrong while running the audit:")
            st.exception(e)
else:
    st.write("Upload a CSV file to begin.")