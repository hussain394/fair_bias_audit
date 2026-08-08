# Fairness & Bias Audit Tool

A tool that trains a classifier, audits it for demographic bias using industry-standard fairness metrics, and applies multiple mitigation strategies — built to generalize to **any uploaded tabular dataset**, not just one.

**Live demo:** https://fair-bias-audit.streamlit.app

## Problem

ML models used in high-stakes decisions (credit approval, hiring, lending) can learn and amplify bias present in historical data, even without being told the sensitive attribute directly. This project builds an audit pipeline to detect that bias with concrete metrics, and to measure how much mitigation strategies help — and what they cost in accuracy.

## What it does

1. Trains a baseline classifier (Logistic Regression or XGBoost) on a dataset
2. Measures group-wise fairness metrics (demographic parity, equalized odds, selection rate, false positive/negative rates) using [Fairlearn](https://fairlearn.org/)
3. Applies mitigation strategies — pre-processing (Correlation Remover), in-processing (Exponentiated Gradient), post-processing (Threshold Optimizer) — and compares accuracy/fairness trade-offs
4. Presents everything in an interactive Streamlit dashboard, including a **universal mode** that accepts any uploaded CSV with dynamic target/sensitive-attribute selection

## Key Results

| Dataset | Sensitive Attribute | Baseline DP Diff | Mitigated DP Diff | Accuracy Cost |
|---|---|---|---|---|
| UCI Adult Income | sex | 0.108 | 0.003 (Exp. Gradient) | ~2% |
| German Credit | sex | 0.122 | 0.013 (Exp. Gradient) | <1% |

*(DP Diff = Demographic Parity Difference; 0 = perfectly fair)*

**Finding:** the fairness-accuracy trade-off is not fixed — it varies significantly by dataset. On German Credit, near-elimination of the bias gap cost under 1% accuracy; on Adult Income, achieving a similar reduction cost roughly 2%.

## Tech Stack

- **ML:** scikit-learn, XGBoost
- **Fairness:** Fairlearn (`MetricFrame`, `ExponentiatedGradient`, `ThresholdOptimizer`, `CorrelationRemover`)
- **Dashboard:** Streamlit
- **Data:** UCI Adult Income, German Credit (Statlog), via OpenML

## Project Structure

```
fairness-bias-audit/
├── data/                       # raw + processed datasets
├── src/
│   ├── load_data.py            # Adult Income loader
│   ├── load_german_credit.py   # German Credit loader
│   ├── eda.py                  # exploratory analysis sliced by sensitive attr
│   ├── train_baseline.py       # baseline model training
│   ├── measure_bias.py         # Fairlearn bias measurement
│   └── mitigate_bias.py        # mitigation strategies + comparison
├── app/
│   ├── dashboard.py             # dashboard for Adult Income (fixed dataset)
│   └── dashboard_universal.py   # generalized: upload any CSV, pick model
├── reports/                    # generated metrics, charts, comparison tables
└── requirements.txt
```

## Run locally

```powershell
git clone https://github.com/YOUR_USERNAME/fairness-bias-audit.git
cd fairness-bias-audit
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Reproduce the pipeline end-to-end
python src\load_data.py
python src\eda.py
python src\train_baseline.py
python src\measure_bias.py
python src\mitigate_bias.py

# Launch the universal dashboard
streamlit run app\dashboard_universal.py
```

## Why this matters

Regulatory frameworks like the EU AI Act and US EEOC guidance on algorithmic hiring/lending increasingly require models used in high-stakes decisions to be auditable for disparate impact. This project demonstrates that full audit workflow — measurement, mitigation, and trade-off reporting — in a reusable, dataset-agnostic form.
