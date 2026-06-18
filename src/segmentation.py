"""
Unsupervised module: customer segmentation.

Builds RFM features (Recency, Frequency, Monetary), scales them, picks k with the
silhouette score, clusters with K-Means, and visualises the segments in 2D with PCA.
Saves the scaler + kmeans + a human-readable segment profile.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import RANDOM_STATE, MODELS, FIGURES
from src.data_prep import build_customer_table

RFM = ["recency_days", "frequency", "monetary"]


def run_segmentation(df=None) -> dict:
    if df is None:
        df = build_customer_table()
    X = df[RFM].copy()
    Xs = StandardScaler().fit(X)
    Xz = Xs.transform(X)

    # choose k by silhouette over a small range
    scores = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xz)
        scores[k] = float(silhouette_score(Xz, km.labels_))
    best_k = max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(Xz)
    df = df.copy()
    df["segment"] = km.labels_

    # profile each segment with average RFM + size, then attach a readable label
    profile = df.groupby("segment")[RFM].mean().round(1)
    profile["customers"] = df.groupby("segment").size()
    profile["avg_clv_target"] = df.groupby("segment")["clv_target"].mean().round(1)
    profile["label"] = [_label(r) for _, r in profile.iterrows()]

    # PCA to 2D for the scatter plot
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xz)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=km.labels_, cmap="tab10", s=8, alpha=.7)
    ax.set_title(f"Customer segments (k={best_k}, PCA view)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); plt.colorbar(sc, label="segment")
    fig.tight_layout(); fig.savefig(FIGURES / "segments_pca.png", dpi=110); plt.close(fig)

    joblib.dump({"scaler": Xs, "kmeans": km, "rfm_cols": RFM},
                MODELS / "segmentation.joblib")
    profile.to_csv(MODELS / "segment_profile.csv")

    return {"best_k": best_k, "silhouette_by_k": scores,
            "profile": profile.reset_index().to_dict(orient="records")}


def _label(row) -> str:
    """Simple, explainable naming rule from each segment's average RFM."""
    if row["recency_days"] > 120:
        return "At-risk / lapsing"
    if row["frequency"] >= 50:
        return "VIP / wholesale"
    if row["frequency"] >= 10:
        return "Champions"
    if row["frequency"] <= 3:
        return "New / low-engagement"
    return "Loyal mid-value"


if __name__ == "__main__":
    import json
    r = run_segmentation()
    print(json.dumps({k: v for k, v in r.items() if k != "profile"}, indent=2))
    print(pd.DataFrame(r["profile"]).to_string(index=False))