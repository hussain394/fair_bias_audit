import pandas as pd
import os

# Resolve path to german_credit_data.csv in the app directory
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "german_credit_data.csv")

df = pd.read_csv(file_path)

print(f"Total Columns: {len(df.columns)}")
print(f"Shape (Rows, Columns): {df.shape}")
print("\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")