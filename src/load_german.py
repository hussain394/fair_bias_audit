# src/load_german_credit.py
from sklearn.datasets import fetch_openml
import pandas as pd
import os

def load_german_credit(save_path="data/german_credit_raw.csv"):
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = data.frame

    # 'class' is the target: 'good' or 'bad' credit risk
    print("Target values:", df["class"].unique())

    # 'personal_status' mixes sex + marital status, e.g. "male single",
    # "female div/dep/mar" — this is a known quirk of this dataset.
    # We split it into a clean 'sex' column so it's usable as a sensitive attribute.
    print("\npersonal_status values:", df["personal_status"].unique())
    df["sex"] = df["personal_status"].apply(lambda x: "female" if "female" in x else "male")

    # 'age' is numeric — bucket it so group metrics are readable
    # (age 25 is the standard cutoff used in fairness literature for this dataset)
    df["age_group"] = df["age"].apply(lambda x: "25 or under" if x <= 25 else "over 25")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")
    print(df[["class", "sex", "age_group"]].head())

    return df

if __name__ == "__main__":
    load_german_credit()