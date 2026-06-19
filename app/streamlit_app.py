"""
Customer Intelligence Engine — Streamlit dashboard.

Loads the artifacts produced by `python -m src.train_all` and exposes five pages:
Overview, Score a customer, Segments, Basket recommender, and Causal impact.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

# make `src` importable no matter where streamlit is launched from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import joblib

from src.config import (MODELS, FIGURES, NUM_FEATURES, CAT_FEATURES, FEATURES)
from src.data_prep import build_customer_table

st.set_page_config(page_title="Customer Intelligence Engine", layout="wide")


# --------------------------------------------------------------------------
# Loaders (cached so the app is snappy)
# --------------------------------------------------------------------------
@st.cache_data
def load_summary():
    p = MODELS / "results_summary.json"
    return json.load(open(p)) if p.exists() else None


@st.cache_data
def load_customers():
    return build_customer_table()


@st.cache_resource
def load_models():
    reg = joblib.load(MODELS / "clv_regressor.joblib")
    clf = joblib.load(MODELS / "churn_classifier.joblib")
    return reg, clf


def fig(name):
    p = FIGURES / name
    if p.exists():
        st.image(str(p), use_container_width=True)


summary = load_summary()
if summary is None:
    st.error("No artifacts found. Run `python -m src.train_all` first.")
    st.stop()

page = st.sidebar.radio(
    "Page",
    ["Overview", "Score a customer", "Segments", "Basket recommender", "Causal impact"],
)
st.sidebar.caption("Built on the DSI classical-ML toolkit · real Online Retail II data")


# --------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------
if page == "Overview":
    st.title("Customer Intelligence Engine")
    st.write("One dataset, five questions: who is valuable, who will churn, how "
             "customers cluster, what sells together, and what a promo really did.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{summary['dataset']['customers']:,}")
    c2.metric("Churn rate", f"{summary['dataset']['churn_rate']*100:.1f}%")
    c3.metric("Churn model AUC", summary["classification"]["RandomForest_tuned"]["roc_auc"])

    st.subheader("Business overview")
    a, b = st.columns(2)
    with a:
        fig("revenue_over_time.png")
        fig("top_products.png")
    with b:
        fig("monetary_hist.png")
        fig("revenue_by_country.png")

    st.subheader("Model scorecards")
    a, b = st.columns(2)
    with a:
        st.markdown("**CLV regression (test set)**")
        st.dataframe(pd.DataFrame(summary["regression"]).T, use_container_width=True)
    with b:
        st.markdown("**Churn classification (test set)**")
        st.dataframe(pd.DataFrame(summary["classification"]).T, use_container_width=True)

    st.subheader("Hypothesis tests")
    st.json(summary["stats"])


# --------------------------------------------------------------------------
# 2. Score a customer
# --------------------------------------------------------------------------
elif page == "Score a customer":
    st.title("Score a customer")
    st.write("Enter a customer's behaviour and the models predict their future "
             "spend and churn risk. (Both models are sklearn Pipelines, so they "
             "handle scaling and encoding internally.)")

    cust = load_customers()
    reg, clf = load_models()
    med = cust[NUM_FEATURES].median()

    c1, c2, c3 = st.columns(3)
    inp = {}
    inp["recency_days"] = c1.number_input("Recency (days since last order)", 0, 365, int(med["recency_days"]))
    inp["frequency"] = c1.number_input("Frequency (orders)", 1, 200, int(med["frequency"]))
    inp["monetary"] = c2.number_input("Monetary (total spend)", 0.0, 1e5, float(med["monetary"]))
    inp["num_items"] = c2.number_input("Total items bought", 1, 5000, int(med["num_items"]))
    inp["distinct_products"] = c3.number_input("Distinct products", 1, 200, int(med["distinct_products"]))
    inp["tenure_days"] = c3.number_input("Tenure (days as a customer)", 0, 1000, int(med["tenure_days"]))
    inp["spend_rank"] = st.slider("Spend rank (1 = top spender)", 1, int(cust["spend_rank"].max()),
                                  int(med["spend_rank"]))
    inp["country"] = st.selectbox("Country", sorted(cust["country"].unique()))
    inp["avg_order_value"] = inp["monetary"] / max(inp["frequency"], 1)  # derived

    row = pd.DataFrame([{k: inp[k] for k in FEATURES}])

    if st.button("Predict", type="primary"):
        clv = float(reg.predict(row)[0])
        proba = float(clf["pipeline"].predict_proba(row)[0, 1])
        churns = proba >= clf["threshold"]
        c1, c2 = st.columns(2)
        c1.metric("Predicted next-period spend", f"{clv:,.0f}")
        c2.metric("Churn probability", f"{proba*100:.1f}%",
                  delta="at risk" if churns else "likely to stay",
                  delta_color="inverse" if churns else "normal")
        st.caption(f"Decision threshold = {clf['threshold']:.2f} (tuned for best F1).")
        if churns and clv > cust["clv_target"].median():
            st.warning("High value **and** high churn risk — prioritise this customer "
                       "for a retention offer.")


# --------------------------------------------------------------------------
# 3. Segments
# --------------------------------------------------------------------------
elif page == "Segments":
    st.title("Customer segments")
    st.write("RFM features clustered with K-Means, visualised with PCA. Each "
             "segment gets a marketing action.")
    fig("segments_pca.png")

    prof_path = MODELS / "segment_profile.csv"
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        st.dataframe(prof, use_container_width=True)

    st.subheader("Suggested actions")
    actions = {
        "VIP / wholesale": "Account-manage — dedicated contact, volume pricing, priority stock.",
        "Champions": "Reward and upsell — loyalty perks, early access, referral asks.",
        "Loyal mid-value": "Grow basket size — bundles and cross-sell (see Basket recommender).",
        "At-risk / lapsing": "Win-back — targeted discount before they're gone for good.",
        "New / low-engagement": "Onboard — welcome series, education, a second-purchase nudge.",
    }
    for label, action in actions.items():
        st.markdown(f"- **{label}** — {action}")


# --------------------------------------------------------------------------
# 4. Basket recommender
# --------------------------------------------------------------------------
elif page == "Basket recommender":
    st.title("Basket recommender")
    st.write("Association rules mined with Apriori. Pick a product to see what is "
             "frequently bought with it (ranked by lift).")

    rules_path = MODELS / "association_rules.csv"
    if not rules_path.exists() or pd.read_csv(rules_path).empty:
        st.info("No association rules were found. With the real dataset, lower "
                "min_support in src/association_rules.py.")
    else:
        rules = pd.read_csv(rules_path)
        products = sorted(set(rules["antecedents"]))
        pick = st.selectbox("If a customer buys…", products)
        recs = rules[rules["antecedents"] == pick].sort_values("lift", ascending=False)
        st.markdown("**Recommend:**")
        st.dataframe(recs[["consequents", "confidence", "lift"]].reset_index(drop=True),
                     use_container_width=True)


# --------------------------------------------------------------------------
# 5. Causal impact
# --------------------------------------------------------------------------
elif page == "Causal impact":
    st.title("Causal impact of the promotion")
    st.write("What did the promo really add, versus what revenue *would have been* "
             "without it (the counterfactual)?")
    ci = summary["causal_impact"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual revenue", f"{ci['actual_revenue']:,.0f}")
    c2.metric("Expected (no promo)", f"{ci['expected_revenue']:,.0f}")
    c3.metric("Estimated lift", f"{ci['estimated_lift']:,.0f}",
              delta=f"{ci['relative_lift_pct']}%")
    fig("causal_impact.png")
    st.caption(f"Event window: {ci['event_window'][0]} → {ci['event_window'][1]}. "
               "Production version: swap in the CausalImpact library for Bayesian "
               "structural time series.")