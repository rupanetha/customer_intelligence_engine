"""
Exploratory data analysis. Saves figures to reports/figures/ so the README and
the Streamlit app can show them. Uses matplotlib only (the DSI plotting library).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # render to files without a display
import matplotlib.pyplot as plt

from src.config import FIGURES
from src.data_prep import load_transactions, build_customer_table


def run_eda() -> dict:
    tx = load_transactions()
    cust = build_customer_table()
    paths = {}

    # 1. Daily revenue over time
    daily = tx.set_index("InvoiceDate")["Revenue"].resample("D").sum()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily.index, daily.values, lw=0.9)
    ax.set_title("Daily revenue"); ax.set_ylabel("Revenue"); ax.grid(alpha=.3)
    fig.tight_layout(); p = FIGURES / "revenue_over_time.png"; fig.savefig(p, dpi=110); plt.close(fig)
    paths["revenue_over_time"] = p

    # 2. Distribution of customer monetary value
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(cust["monetary"], bins=40, color="#3b6ea5")
    ax.set_title("Customer value (feature period)"); ax.set_xlabel("Monetary"); ax.set_ylabel("Customers")
    fig.tight_layout(); p = FIGURES / "monetary_hist.png"; fig.savefig(p, dpi=110); plt.close(fig)
    paths["monetary_hist"] = p

    # 3. Top 10 products by revenue
    top = (tx.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(top.index[::-1], top.values[::-1], color="#4a7a4a")
    ax.set_title("Top 10 products by revenue"); ax.set_xlabel("Revenue")
    fig.tight_layout(); p = FIGURES / "top_products.png"; fig.savefig(p, dpi=110); plt.close(fig)
    paths["top_products"] = p

    # 4. Revenue by country
    by_c = tx.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(by_c.index, by_c.values, color="#d97724")
    ax.set_title("Revenue by country"); ax.set_ylabel("Revenue")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); p = FIGURES / "revenue_by_country.png"; fig.savefig(p, dpi=110); plt.close(fig)
    paths["revenue_by_country"] = p

    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    print("Saved:", run_eda())