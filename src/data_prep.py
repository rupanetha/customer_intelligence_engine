
from __future__ import annotations
import sqlite3
import pandas as pd

from src.config import (RAW_FILE, HOLDOUT_DAYS, NUM_FEATURES, CAT_FEATURES,
                        CLV_TARGET, CHURN_TARGET, RANDOM_STATE)

_TX_CACHE = None  # load once per process


# --------------------------------------------------------------------------
# 1. Load raw transactions (real file if present, else synthetic)
# --------------------------------------------------------------------------
def load_transactions(force_reload: bool = False) -> pd.DataFrame:
    global _TX_CACHE
    if _TX_CACHE is not None and not force_reload:
        return _TX_CACHE.copy()

    if RAW_FILE.exists():
        df = pd.read_csv(RAW_FILE)
    else:
        from src.generate_synthetic import generate
        print("No real dataset found -> using synthetic data. Put online_retail_II.csv "
              "in data/raw/ to use the real data.")
        df = generate()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["Customer ID"])                       # drop anonymous rows
    df = df[~df["Invoice"].astype(str).str.startswith("C")]      # drop cancellations
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]            # drop returns / bad rows
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["Revenue"] = df["Quantity"] * df["Price"]                 # derived line revenue
    df = df.reset_index(drop=True)

    _TX_CACHE = df
    return df.copy()


# --------------------------------------------------------------------------
# 2. Build customer features with SQL, split on time to avoid target leakage
# --------------------------------------------------------------------------
def build_customer_table() -> pd.DataFrame:
    tx = load_transactions()
    split_date = tx["InvoiceDate"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    split_str = split_date.strftime("%Y-%m-%d %H:%M:%S")

    tx_sql = tx.copy()
    tx_sql["InvoiceDate"] = tx_sql["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")

    con = sqlite3.connect(":memory:")
    tx_sql.to_sql("transactions", con, index=False)


    feature_sql = """
    WITH feat AS (
        SELECT * FROM transactions WHERE InvoiceDate < :split
    ),
    agg AS (
        SELECT
            "Customer ID"                      AS customer_id,
            MIN(Country)                       AS country,
            COUNT(DISTINCT Invoice)            AS frequency,
            SUM(Revenue)                       AS monetary,
            SUM(Quantity)                      AS num_items,
            COUNT(DISTINCT StockCode)          AS distinct_products,
            MIN(InvoiceDate)                   AS first_purchase,
            MAX(InvoiceDate)                   AS last_purchase
        FROM feat
        GROUP BY "Customer ID"
    )
    SELECT
        a.*,
        a.monetary * 1.0 / a.frequency               AS avg_order_value,
        RANK() OVER (ORDER BY a.monetary DESC)       AS spend_rank
    FROM agg a
    """
    feats = pd.read_sql_query(feature_sql, con, params={"split": split_str})

  
    holdout_sql = """
        SELECT "Customer ID"            AS customer_id,
               SUM(Revenue)             AS clv_target,
               COUNT(DISTINCT Invoice)  AS holdout_orders
        FROM transactions
        WHERE InvoiceDate >= :split
        GROUP BY "Customer ID"
    """
    hold = pd.read_sql_query(holdout_sql, con, params={"split": split_str})
    con.close()

   
    df = feats.merge(hold, on="customer_id", how="left")
    df["clv_target"] = df["clv_target"].fillna(0.0)
    df["holdout_orders"] = df["holdout_orders"].fillna(0)
    df["churned"] = (df["holdout_orders"] == 0).astype(int)

    df["first_purchase"] = pd.to_datetime(df["first_purchase"])
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    df["recency_days"] = (split_date - df["last_purchase"]).dt.days
    df["tenure_days"] = (split_date - df["first_purchase"]).dt.days

    # shuffle so rows aren't ordered by spend (the SQL RANK orders them); this keeps
    # cross-validation folds representative later on.
    df = clean_customer_table(df)
    return df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Clean the customer table (missing values + outlier capping)
# --------------------------------------------------------------------------
def clean_customer_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["monetary", "avg_order_value", "num_items", "clv_target"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)
    for col in NUM_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df["country"] = df["country"].fillna("Unknown")
    return df


def get_modeling_frames():
    return build_customer_table(), NUM_FEATURES, CAT_FEATURES


if __name__ == "__main__":
    d = build_customer_table()
    print(d[NUM_FEATURES + CAT_FEATURES + [CLV_TARGET, CHURN_TARGET]].describe(include="all"))
    print("\nChurn rate:", round(d[CHURN_TARGET].mean(), 3), "| customers:", len(d))

# ----------------------------------------------------------------------------
    
    
    
    