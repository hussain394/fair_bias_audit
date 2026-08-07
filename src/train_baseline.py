# src/train_baseline.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

def load_and_prepare(data_path="data/adult_income_raw.csv"):
    df = pd.read_csv(data_path)
    df = df.dropna()  # simple handling for now; revisit if needed

    # Keep sensitive attributes aside BEFORE encoding, so we can slice
    # fairness metrics by them later without decoding anything
    sensitive = df[["sex", "race"]].copy()

    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income"])

    # Encode categoricals (simple label encoding is fine for a baseline;
    # we're not optimizing accuracy here, we're building the audit pipeline)
    cat_cols = X.select_dtypes(include="category").columns.tolist() + \
               X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y, sensitive

def train_and_evaluate():
    X, y, sensitive = load_and_prepare()

    # Split, keeping sensitive attributes aligned with the same indices
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results.append({"model": name, "accuracy": acc, "f1": f1})

        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds))

        joblib.dump(model, f"reports/{name}_baseline.pkl")

    pd.DataFrame(results).to_csv("reports/baseline_results.csv", index=False)

    # Save test split + predictions + sensitive attrs together —
    # Phase 2 (fairness measurement) needs exactly this combination
    test_export = X_test.copy()
    test_export["y_true"] = y_test.values
    test_export["sex"] = sens_test["sex"].values
    test_export["race"] = sens_test["race"].values
    for name, model in models.items():
        test_export[f"pred_{name}"] = model.predict(X_test)
    test_export.to_csv("data/splits/test_with_predictions.csv", index=False)

    print("\nSaved: reports/baseline_results.csv, models (.pkl), and test predictions.")

if __name__ == "__main__":
    train_and_evaluate()