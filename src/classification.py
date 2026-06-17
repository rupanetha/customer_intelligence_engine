"""
Classification module: predict churn (no purchase in the holdout period).

Covers: logistic regression, classification tree, random forest, KNN,
confusion matrix, precision/recall/F1, ROC-AUC, threshold tuning, GridSearchCV.
Saves the best model (plus its tuned threshold) as a Pipeline.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.pipeline import Pipeline

from src.config import (NUM_FEATURES, CAT_FEATURES, FEATURES, CHURN_TARGET,
                        RANDOM_STATE, MODELS, FIGURES)
from src.model_utils import make_preprocessor
from src.data_prep import build_customer_table


def run_classification(df=None) -> dict:
    if df is None:
        df = build_customer_table()
    X, y = df[FEATURES], df[CHURN_TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE,
                                          stratify=y)
    pre = make_preprocessor(NUM_FEATURES, CAT_FEATURES)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                               n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }
    results = {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        results[name] = _metrics(yte, (proba >= 0.5).astype(int), proba)

    grid = GridSearchCV(
        Pipeline([("pre", pre), ("model", RandomForestClassifier(random_state=RANDOM_STATE,
                                                                 n_jobs=-1))]),
        param_grid={
            "model__n_estimators": [200],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [2],
        },
        cv=3, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    proba = best.predict_proba(Xte)[:, 1]
    res = _metrics(yte, (proba >= 0.5).astype(int), proba)
    res["best_params"] = grid.best_params_
    results["RandomForest_tuned"] = res

    # threshold tuning: pick the probability cut-off that maximises F1
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(yte, (proba >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    results["threshold_tuning"] = {"best_threshold": round(best_t, 3),
                                   "f1_at_best": round(best_f1, 3)}

    _save_roc(yte, proba)
    _save_confusion(yte, (proba >= best_t).astype(int))
    joblib.dump({"pipeline": best, "threshold": best_t}, MODELS / "churn_classifier.joblib")
    return results


def _metrics(y_true, y_pred, proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 3),
    }


def _save_roc(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#3b6ea5", label=f"AUC={roc_auc_score(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Churn model - ROC curve"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "churn_roc.png", dpi=110); plt.close(fig)


def _save_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["stay", "churn"]); ax.set_yticklabels(["stay", "churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(FIGURES / "churn_confusion.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    import json
    print(json.dumps(run_classification(), indent=2))