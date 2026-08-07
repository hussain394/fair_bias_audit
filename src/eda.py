# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(data_path="data/adult_income_raw.csv", out_dir="reports/eda"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nTarget distribution:\n", df["income"].value_counts(normalize=True))

    # --- The core fairness-relevant EDA: outcome rate by sensitive attribute ---
    for attr in ["sex", "race"]:
        rate = df.groupby(attr)["income"].apply(
            lambda x: (x == ">50K").mean()
        ).sort_values(ascending=False)
        print(f"\n>50K income rate by {attr}:\n", rate)

        plt.figure(figsize=(7, 4))
        sns.barplot(x=rate.index, y=rate.values)
        plt.title(f"Proportion earning >50K by {attr}")
        plt.ylabel("Proportion >50K")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/income_rate_by_{attr}.png")
        plt.close()

    # Correlation heatmap for numeric features (sanity check, not fairness-specific)
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png")
    plt.close()

    print(f"\nCharts saved to {out_dir}/")

if __name__ == "__main__":
    run_eda()