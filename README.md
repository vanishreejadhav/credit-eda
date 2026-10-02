# Credit Risk Exploratory Data Analysis

GitHub repository: `credit-eda`.

A reproducible Python EDA project for the consumer credit risk dataset. The analysis is designed to identify patterns associated with loan default and prepare a defensible feature review for subsequent probability-of-default (PD) modeling. It does not train a model or interpret associations as causal effects.

## Dataset

The analysis expects `data/credit_risk_dataset.csv` with `loan_status` as the target (`1` = default, `0` = non-default). Predictors cover borrower demographics and employment, home ownership, loan purpose and terms, credit grade, prior default history, and credit-history length. The input file is not modified by the EDA.

The supplied dataset has 32,581 rows and 12 columns. Initial profiling found 4,011 missing cells, 165 duplicate rows, and 7,108 defaults (21.8%). The first records also include a 123-year employment length, which the quality checks flag for review. These figures are dataset-level observations, not claims about causal risk drivers. Run the script to regenerate the detailed findings and confirm all counts.

## Analysis and outputs

`credit_risk_eda.py` provides reusable functions and a command-line entry point. It creates `outputs/eda/eda_summary.xlsx` and PNGs for missingness, target balance, univariate distributions, default rates by category, and numeric correlations. The workbook includes dataset overview, target balance, missing/domain checks, IQR and z-score outliers, skewness, category default rates, chi-square and Mann-Whitney U tests, numeric correlations, VIF, ranked WoE/IV, and concise findings.

WoE uses the convention `ln(non-default distribution / default distribution)` with 0.5 count smoothing; numeric predictors with more than 10 distinct observed values are binned into quantiles. IV labels are weak (<0.02), medium (0.02 to <0.10), and strong (>=0.10). Significance tests are unadjusted and should be interpreted in light of sample size and multiple comparisons. VIF is computed for numeric predictors only.

The script flags impossible/nonpositive values according to simple domain checks, including age above 100 and employment length above 60. Outlier flags are screening signals, not automatic exclusions. Missing-value handling, duplicate treatment, feature selection, and time/validation design should be decided explicitly before PD model training.

## Run

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python credit_risk_eda.py
```

To provide another dataset path or output directory:

```powershell
python credit_risk_eda.py --input data/credit_risk_dataset.csv --output-dir outputs/eda
```

If the CSV is not under `data/`, pass its actual path with `--input`. The runner also checks common workspace locations, including the nested `credit_risk_dataset.csv/credit_risk_dataset.csv` layout.
