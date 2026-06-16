"""
Statistics / A/B-testing module (DSI hypothesis-testing + A/B lessons).

Two classic tests on real business questions:
  1. t-test     : do UK customers have a different average order value than non-UK?
  2. chi-square : is country (UK vs rest) associated with churn?
"""
from __future__ import annotations
from scipy import stats
import pandas as pd

from src.data_prep import build_customer_table


def run_stats_tests(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = build_customer_table()
    out = {}

    # --- 1. Independent-samples t-test: AOV, UK vs non-UK ---
    uk = df.loc[df["country"] == "United Kingdom", "avg_order_value"]
    other = df.loc[df["country"] != "United Kingdom", "avg_order_value"]
    t_stat, t_p = stats.ttest_ind(uk, other, equal_var=False)  # Welch's t-test
    out["ttest_aov_uk_vs_other"] = {
        "uk_mean": round(float(uk.mean()), 2),
        "other_mean": round(float(other.mean()), 2),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(t_p), 4),
        "significant_at_0.05": bool(t_p < 0.05),
    }

    # --- 2. Chi-square test of independence: UK flag vs churn ---
    df = df.copy()
    df["is_uk"] = (df["country"] == "United Kingdom").astype(int)
    table = pd.crosstab(df["is_uk"], df["churned"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(table)
    out["chisq_country_vs_churn"] = {
        "chi2": round(float(chi2), 3),
        "p_value": round(float(chi_p), 4),
        "dof": int(dof),
        "significant_at_0.05": bool(chi_p < 0.05),
    }
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_stats_tests(), indent=2))