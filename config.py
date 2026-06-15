
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
for _p in (DATA_RAW, MODELS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


RAW_FILE = DATA_RAW / "online_retail_II.csv"

RANDOM_STATE = 42

HOLDOUT_DAYS = 90

NUM_FEATURES = [
    "recency_days",        # days from last purchase to the split date
    "frequency",           # number of distinct orders in the feature period
    "monetary",            # total spend in the feature period
    "avg_order_value",     # monetary / frequency
    "num_items",           # total quantity bought
    "distinct_products",   # how many different products
    "tenure_days",         # days from first purchase to the split date
    "spend_rank",          # SQL window-function rank by monetary (1 = top spender)
]
CAT_FEATURES = ["country"]
FEATURES = NUM_FEATURES + CAT_FEATURES

CLV_TARGET = "clv_target"     # spend in the holdout period (regression target)
CHURN_TARGET = "churned"      # 1 = no purchase in the holdout period (classification target)