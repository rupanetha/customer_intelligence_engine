"""
Regression module: predict each customer's future spend (CLV proxy = spend in the
holdout period) from their past behaviour.

Covers: linear regression, regression tree, random forest, R^2 / MAE / RMSE,
cross-validation, hyperparameter tuning with GridSearchCV, and feature importance.
Saves the best model as a Pipeline so the app can predict from raw inputs.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.config import (NUM_FEATURES, CAT_FEATURES, FEATURES, CLV_TARGET,
                        RANDOM_STATE, MODELS, FIGURES)
from src.model_utils import make_preprocessor
from src.data_prep import build_customer_table


def run_regression(df=None) -> dict:
    if df is None:
        df = build_customer_table()
    X, y = df[FEATURES], df[CLV_TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    pre = make_preprocessor(NUM_FEATURES, CAT_FEATURES)

    # compare three model families on the held-out test set
    candidates = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE,
                                              n_jobs=-1),
    }
    results = {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = _metrics(yte, pred)

    # tune the random forest with a small, sensible grid (cv=3 keeps it quick)
    grid = GridSearchCV(
        Pipeline([("pre", pre), ("model", RandomForestRegressor(random_state=RANDOM_STATE,
                                                                n_jobs=-1))]),
        param_grid={
            "model__n_estimators": [200],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [2],
        },
        cv=3, scoring="r2", n_jobs=-1,
    )
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    pred = best.predict(Xte)
    res = _metrics(yte, pred)
    res["best_params"] = grid.best_params_
    res["cv_r2"] = round(float(cross_val_score(best, X, y, cv=3, scoring="r2").mean()), 3)
    results["RandomForest_tuned"] = res

    _save_feature_importance(best)
    joblib.dump(best, MODELS / "clv_regressor.joblib")
    return results


def _metrics(y_true, pred) -> dict:
    return {
        "r2": round(float(r2_score(y_true, pred)), 3),
        "mae": round(float(mean_absolute_error(y_true, pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 2),
    }


def _save_feature_importance(pipe: Pipeline):
    pre = pipe.named_steps["pre"]
    names = list(NUM_FEATURES) + list(
        pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CAT_FEATURES)
    )
    imp = pipe.named_steps["model"].feature_importances_
    order = np.argsort(imp)[-12:]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in order], imp[order], color="#3b6ea5")
    ax.set_title("CLV model - feature importance"); fig.tight_layout()
    fig.savefig(FIGURES / "regression_importance.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    import json
    print(json.dumps(run_regression(), indent=2))