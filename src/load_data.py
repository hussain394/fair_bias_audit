# src/load_data.py
from sklearn.datasets import fetch_openml
import pandas as pd
import os

def load_adult_income(save_path="data/adult_income_raw.csv"):
    # as_frame=True gives us a pandas DataFrame directly, version=2 is the
    # cleaned "adult" dataset on OpenML with consistent column names
    data = fetch_openml(name="adult", version=2, as_frame=True)

    df = data.frame  # features + target combined
    df.rename(columns={"class": "income"}, inplace=True)  # target column is called 'class'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved to {save_path}")
    print("\nSensitive attribute value counts:")
    print(df["sex"].value_counts())
    print(df["race"].value_counts())

    return df

if __name__ == "__main__":
    load_adult_income()