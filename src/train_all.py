"""
Orchestrator: runs the whole analysis end-to-end and saves every artifact the
Streamlit app needs. Run this ONCE before launching the app:

    python -m src.train_all
"""
from __future__ import annotations
import json
import time

from src.config import MODELS
from src.data_prep import build_customer_table
from src.eda import run_eda
from src.stats_tests import run_stats_tests
from src.regression import run_regression
from src.classification import run_classification
from src.segmentation import run_segmentation
from src.association_rules import run_association_rules
from src.causal_impact import run_causal_impact


def main():
    t0 = time.time()
    summary = {}

    print("1/7  Building customer table ..."); df = build_customer_table()
    summary["dataset"] = {"customers": int(len(df)),
                          "churn_rate": round(float(df["churned"].mean()), 3)}

    print("2/7  EDA figures ...");           summary["eda_figures"] = run_eda()
    print("3/7  Hypothesis tests ...");      summary["stats"] = run_stats_tests(df)
    print("4/7  Regression (CLV) ...");      summary["regression"] = run_regression(df)
    print("5/7  Classification (churn) ..."); summary["classification"] = run_classification(df)
    print("6/7  Segmentation ...");          summary["segmentation"] = run_segmentation(df)
    print("7/7  Association rules + causal impact ...")
    summary["association"] = run_association_rules()
    summary["causal_impact"] = run_causal_impact()

    with open(MODELS / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts written to {MODELS}/")
    print("Launch the dashboard with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

