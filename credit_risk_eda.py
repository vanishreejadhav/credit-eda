"""Exploratory analysis for the consumer credit risk dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET = "loan_status"
NUMERIC_LIMITS = {
    "person_age": (0, 100),
    "person_income": (0, None),
    "person_emp_length": (0, 60),
    "loan_amnt": (0, None),
    "loan_int_rate": (0, None),
    "loan_percent_income": (0, None),
    "cb_person_cred_hist_length": (0, None),
}
EXPECTED_COLUMNS = {
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
}
OUTPUT_DIR = Path("outputs") / "eda"
LOGGER = logging.getLogger("credit_risk_eda")


def resolve_input_path(input_path: str | Path | None = None) -> Path:
    """Resolve an explicit path or a common project/workspace dataset location."""
    if input_path is not None:
        path = Path(input_path).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Dataset not found: {path}")

    base = Path(__file__).resolve().parent
    candidates = [
        Path("data/credit_risk_dataset.csv"),
        base / "data" / "credit_risk_dataset.csv",
        Path("credit_risk_dataset.csv"),
        Path("credit_risk_dataset.csv") / "credit_risk_dataset.csv",
        base / "credit_risk_dataset.csv" / "credit_risk_dataset.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate credit_risk_dataset.csv. Checked: {locations}")


def load_data(input_path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV and validate the target and expected input fields."""
    path = resolve_input_path(input_path)
    data = pd.read_csv(path)
    missing_columns = EXPECTED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")
    if not data[TARGET].dropna().isin([0, 1]).all():
        raise ValueError("loan_status must contain only 0/1 values (or missing values).")
    LOGGER.info("Loaded %s rows and %s columns from %s", *data.shape, path)
    return data


def data_overview(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize dimensions, types, nulls, duplicates, and target counts."""
    overview = pd.DataFrame(
        {
            "dtype": data.dtypes.astype(str),
            "missing_count": data.isna().sum(),
            "missing_percent": data.isna().mean().mul(100).round(2),
            "unique_count": data.nunique(dropna=True),
        }
    ).rename_axis("column").reset_index()
    metrics = pd.DataFrame(
        [
            {"metric": "rows", "value": len(data)},
            {"metric": "columns", "value": data.shape[1]},
            {"metric": "duplicate_rows", "value": int(data.duplicated().sum())},
            {"metric": "missing_cells", "value": int(data.isna().sum().sum())},
        ]
    )
    return overview, metrics


def analyze_data_quality(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot missingness and quantify domain-rule and statistical outliers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_rows: list[dict[str, Any]] = []
    for column, (minimum, maximum) in NUMERIC_LIMITS.items():
        if column not in data:
            continue
        values = data[column]
        invalid = values.notna() & ((values <= minimum) if minimum == 0 else (values < minimum))
        if maximum is not None:
            invalid |= values.notna() & (values > maximum)
        quality_rows.append(
            {
                "column": column,
                "rule": f"value > {minimum}" + (f" and <= {maximum}" if maximum else ""),
                "invalid_count": int(invalid.sum()),
                "invalid_percent": round(float(invalid.mean() * 100), 3),
            }
        )

    outlier_rows: list[dict[str, Any]] = []
    numeric_columns = data.select_dtypes(include=np.number).columns.drop(TARGET, errors="ignore")
    for column in numeric_columns:
        values = data[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        standard_deviation = values.std(ddof=0)
        z_count = int((((values - values.mean()).abs() / standard_deviation) > 3).sum()) if standard_deviation else 0
        outlier_rows.append(
            {
                "column": column,
                "iqr_outlier_count": iqr_count,
                "iqr_outlier_percent": round(iqr_count / len(values) * 100, 3),
                "zscore_gt_3_count": z_count,
                "zscore_gt_3_percent": round(z_count / len(values) * 100, 3),
            }
        )

    try:
        plt.figure(figsize=(12, 6))
        sns.heatmap(data.isna(), cbar=False, yticklabels=False, cmap="mako")
        plt.title("Missing value map")
        plt.xlabel("Features")
        plt.tight_layout()
        plt.savefig(output_dir / "missing_values_heatmap.png", dpi=150)
        plt.close()
    except (ValueError, OSError) as exc:
        LOGGER.warning("Could not create missingness heatmap: %s", exc)

    return pd.DataFrame(quality_rows), pd.DataFrame(outlier_rows)


def analyze_target(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Report target counts/rates and save a class-balance plot."""
    counts = data[TARGET].value_counts(dropna=False).sort_index()
    result = pd.DataFrame(
        {
            "loan_status": counts.index.astype(str),
            "count": counts.values,
            "percent": (counts.values / len(data) * 100).round(3),
        }
    )
    try:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=data, x=TARGET, color="#2a788e")
        plt.title("Loan status class balance")
        plt.xlabel("Loan status (1 = default)")
        plt.ylabel("Borrower count")
        plt.tight_layout()
        plt.savefig(output_dir / "target_class_balance.png", dpi=150)
        plt.close()
    except (ValueError, OSError) as exc:
        LOGGER.warning("Could not create target plot: %s", exc)
    return result


def univariate_analysis(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Save numeric distributions/boxplots and categorical count charts."""
    records: list[dict[str, Any]] = []
    numeric_columns = data.select_dtypes(include=np.number).columns.drop(TARGET, errors="ignore")
    categorical_columns = [column for column in data.columns if column not in numeric_columns and column != TARGET]

    for column in numeric_columns:
        values = data[column].dropna()
        if values.empty:
            continue
        records.append(
            {
                "feature": column,
                "kind": "numeric",
                "skewness": float(values.skew()),
                "unique_count": int(values.nunique()),
            }
        )
        try:
            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(values, kde=True, ax=axes[0], color="#2a788e")
            axes[0].set_title(f"{column} distribution")
            sns.boxplot(x=values, ax=axes[1], color="#f6bd60")
            axes[1].set_title(f"{column} boxplot")
            figure.tight_layout()
            figure.savefig(output_dir / f"univariate_{column}.png", dpi=150)
            plt.close(figure)
        except (ValueError, OSError) as exc:
            LOGGER.warning("Could not plot %s: %s", column, exc)

    for column in categorical_columns:
        counts = data[column].fillna("Missing").astype(str).value_counts().head(25)
        records.append(
            {
                "feature": column,
                "kind": "categorical",
                "skewness": np.nan,
                "unique_count": int(data[column].nunique(dropna=True)),
            }
        )
        try:
            figure, axis = plt.subplots(figsize=(9, max(4, len(counts) * 0.3)))
            sns.barplot(x=counts.values, y=counts.index, ax=axis, color="#f6bd60")
            axis.set(title=f"{column} counts", xlabel="Borrower count", ylabel=column)
            figure.tight_layout()
            figure.savefig(output_dir / f"univariate_{column}.png", dpi=150)
            plt.close(figure)
        except (ValueError, OSError) as exc:
            LOGGER.warning("Could not plot %s: %s", column, exc)
    return pd.DataFrame(records)


def bivariate_analysis(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare default rates by category and run appropriate association tests."""
    category_rows: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    features = [column for column in data.columns if column != TARGET]
    for column in features:
        clean = data[[column, TARGET]].dropna()
        if clean.empty:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            default_values = clean.loc[clean[TARGET] == 1, column]
            nondefault_values = clean.loc[clean[TARGET] == 0, column]
            if len(default_values) and len(nondefault_values):
                statistic, p_value = mannwhitneyu(default_values, nondefault_values, alternative="two-sided")
                test_rows.append(
                    {"feature": column, "test": "Mann-Whitney U", "statistic": statistic, "p_value": p_value,
                     "sample_size": len(clean)}
                )
            continue

        clean = clean.copy()
        clean[column] = clean[column].fillna("Missing").astype(str)
        rates = clean.groupby(column, observed=False)[TARGET].agg(default_rate="mean", count="size").reset_index()
        rates.insert(0, "feature", column)
        rates = rates.rename(columns={column: "category"})
        category_rows.append(rates)
        contingency = pd.crosstab(clean[column], clean[TARGET])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            statistic, p_value, _, _ = chi2_contingency(contingency)
            test_rows.append(
                {"feature": column, "test": "Chi-square", "statistic": statistic, "p_value": p_value,
                 "sample_size": len(clean)}
            )
        try:
            figure, axis = plt.subplots(figsize=(9, max(4, len(rates) * 0.35)))
            sns.barplot(data=rates, x="default_rate", y="category", ax=axis, color="#d1495b")
            axis.set(title=f"Default rate by {column}", xlabel="Default rate", ylabel=column)
            axis.set_xlim(0, max(0.1, min(1.0, float(rates["default_rate"].max()) * 1.15)))
            figure.tight_layout()
            figure.savefig(output_dir / f"default_rate_{column}.png", dpi=150)
            plt.close(figure)
        except (ValueError, OSError) as exc:
            LOGGER.warning("Could not plot default rate for %s: %s", column, exc)

    return (
        pd.concat(category_rows, ignore_index=True) if category_rows else pd.DataFrame(),
        pd.DataFrame(test_rows).sort_values("p_value") if test_rows else pd.DataFrame(),
    )


def correlation_and_vif(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot Pearson correlations and estimate VIF for numeric predictors."""
    numeric = data.select_dtypes(include=np.number).drop(columns=[TARGET], errors="ignore")
    correlation = numeric.corr()
    try:
        figure, axis = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
        axis.set_title("Numeric feature correlation (Pearson)")
        figure.tight_layout()
        figure.savefig(output_dir / "correlation_heatmap.png", dpi=150)
        plt.close(figure)
    except (ValueError, OSError) as exc:
        LOGGER.warning("Could not create correlation heatmap: %s", exc)

    predictors = numeric.replace([np.inf, -np.inf], np.nan)
    predictors = predictors.fillna(predictors.median()).dropna(axis=1, how="all")
    predictors = predictors.loc[:, predictors.nunique(dropna=True) > 1]
    vif_rows: list[dict[str, Any]] = []
    if predictors.shape[1] >= 2:
        matrix = predictors.to_numpy(dtype=float)
        for index, column in enumerate(predictors.columns):
            try:
                vif_rows.append({"feature": column, "vif": variance_inflation_factor(matrix, index)})
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
                LOGGER.warning("Could not calculate VIF for %s: %s", column, exc)
                vif_rows.append({"feature": column, "vif": np.nan})
    return correlation.reset_index(names="feature"), pd.DataFrame(vif_rows)


def calculate_woe_iv(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate smoothed WoE/IV per predictor; positive event means default."""
    rows: list[dict[str, Any]] = []
    for column in data.columns.drop(TARGET):
        values = data[column]
        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 10:
            try:
                groups = pd.qcut(values, q=10, duplicates="drop")
                bins = groups.astype("object").where(groups.notna(), "Missing").astype(str)
            except ValueError:
                bins = values.fillna("Missing").astype(str)
        else:
            bins = values.fillna("Missing").astype(str)

        counts = pd.DataFrame({"bin": bins, TARGET: data[TARGET]}).dropna(subset=[TARGET])
        grouped = counts.groupby("bin", observed=False)[TARGET].agg(defaults="sum", total="count")
        grouped["nondefaults"] = grouped["total"] - grouped["defaults"]
        bins_count = max(len(grouped), 1)
        default_distribution = (grouped["defaults"] + 0.5) / (grouped["defaults"].sum() + 0.5 * bins_count)
        nondefault_distribution = (grouped["nondefaults"] + 0.5) / (grouped["nondefaults"].sum() + 0.5 * bins_count)
        woe = np.log(nondefault_distribution / default_distribution)
        iv = ((nondefault_distribution - default_distribution) * woe).sum()
        iv_value = float(iv)
        strength = "weak" if iv_value < 0.02 else "medium" if iv_value < 0.1 else "strong"
        rows.append({"feature": column, "iv": iv_value, "strength": strength, "bin_count": len(grouped)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def build_findings(
    data: pd.DataFrame, metrics: pd.DataFrame, quality: pd.DataFrame, tests: pd.DataFrame, iv: pd.DataFrame
) -> pd.DataFrame:
    """Build concise, data-dependent takeaways for the Excel workbook."""
    defaults = data[TARGET].dropna().mean()
    missing = int(data.isna().sum().sum())
    duplicates = int(data.duplicated().sum())
    findings = [
        f"Dataset contains {len(data):,} rows and {data.shape[1]} columns; default rate is {defaults:.1%}.",
        f"Data contains {missing:,} missing cells ({missing / data.size:.1%} of all cells) and {duplicates:,} duplicate rows.",
    ]
    flagged = quality.loc[quality["invalid_count"] > 0] if not quality.empty else pd.DataFrame()
    if not flagged.empty:
        details = ", ".join(f"{row.column}: {row.invalid_count:,}" for row in flagged.itertuples())
        findings.append(f"Domain-rule values requiring review: {details}.")
    if not iv.empty:
        strongest = iv.iloc[0]
        findings.append(f"Highest univariate information value: {strongest['feature']} (IV={strongest['iv']:.3f}, {strongest['strength']}).")
    if not tests.empty:
        significant = tests.loc[tests["p_value"] < 0.05, "feature"].head(5).tolist()
        if significant:
            findings.append("Features with unadjusted bivariate p < 0.05: " + ", ".join(significant) + ".")
    findings.append("Statistical associations are exploratory, unadjusted, and not evidence of causality.")
    return pd.DataFrame({"finding": findings})


def export_summary(output_dir: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    """Write analysis tables to a multi-sheet Excel workbook."""
    path = output_dir / "eda_summary.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return path


def run_eda(input_path: str | Path | None = None, output_dir: str | Path = OUTPUT_DIR) -> Path:
    """Run the complete EDA and return the summary workbook path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data = load_data(input_path)
    overview, metrics = data_overview(data)
    quality, outliers = analyze_data_quality(data, output_path)
    target = analyze_target(data, output_path)
    univariate = univariate_analysis(data, output_path)
    category_rates, tests = bivariate_analysis(data, output_path)
    correlation, vif = correlation_and_vif(data, output_path)
    iv = calculate_woe_iv(data)
    findings = build_findings(data, metrics, quality, tests, iv)
    workbook = export_summary(
        output_path,
        {
            "key_findings": findings,
            "overview": overview,
            "dataset_metrics": metrics,
            "target_balance": target,
            "data_quality": quality,
            "outliers": outliers,
            "univariate": univariate,
            "default_rates": category_rates,
            "significance_tests": tests,
            "correlation": correlation,
            "vif": vif,
            "woe_iv": iv,
        },
    )
    LOGGER.info("EDA complete. Summary saved to %s", workbook)
    return workbook


def main() -> None:
    """Parse command-line options and run the EDA with clear error reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="CSV input path (defaults to common project locations).")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for plots and the Excel summary.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        workbook = run_eda(arguments.input, arguments.output_dir)
        print(f"EDA complete. Workbook: {workbook}")
    except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
        LOGGER.error("EDA could not be completed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
