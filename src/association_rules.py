"""
Association-rule mining (market-basket analysis).

Turns transactions into a basket matrix (one row per invoice, one column per
product, True/False) and mines rules with Apriori, ranked by lift. To stay fast on
the full real dataset (5,000+ products), we keep only the TOP_N most popular
products before building the basket -- standard practice for market-basket analysis.
"""
from __future__ import annotations
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.config import MODELS
from src.data_prep import load_transactions

TOP_N = 120  # keep the most frequent products (keeps the basket matrix small & fast)


def run_association_rules(min_support: float = 0.02, min_lift: float = 1.1) -> dict:
    tx = load_transactions().dropna(subset=["Description"]).copy()
    tx["Description"] = tx["Description"].str.strip()

    top_products = tx["Description"].value_counts().head(TOP_N).index
    tx = tx[tx["Description"].isin(top_products)]

    # basket matrix: invoice x product, boolean "was it in this basket?"
    basket = (tx.groupby(["Invoice", "Description"])["Quantity"].sum()
                .unstack(fill_value=0))
    basket = (basket > 0)

    rules = None
    for sup in (min_support, 0.01, 0.005, 0.002):
        itemsets = apriori(basket, min_support=sup, use_colnames=True)
        if not itemsets.empty:
            r = association_rules(itemsets, metric="lift", min_threshold=min_lift)
            if not r.empty:
                rules = r
                break
    if rules is None or rules.empty:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support",
                                      "confidence", "lift"])
    rules = rules.sort_values("lift", ascending=False)

    tidy = rules.assign(
        antecedents=rules["antecedents"].apply(lambda s: ", ".join(sorted(s))),
        consequents=rules["consequents"].apply(lambda s: ", ".join(sorted(s))),
    )[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)

    tidy.head(200).to_csv(MODELS / "association_rules.csv", index=False)
    return {"n_rules": int(len(tidy)), "top_rules": tidy.head(10).to_dict(orient="records")}


if __name__ == "__main__":
    r = run_association_rules()
    print("rules found:", r["n_rules"])
    print(pd.DataFrame(r["top_rules"]).to_string(index=False))