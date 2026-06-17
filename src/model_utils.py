"""
Shared modeling helpers. The preprocessor handles the DSI data-prep steps
(imputation, scaling, one-hot encoding) INSIDE an sklearn Pipeline, so the exact
same transforms are applied at train time and at prediction time in the app.
"""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def make_preprocessor(num_features, cat_features) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, num_features),
        ("cat", categorical, cat_features),
    ])