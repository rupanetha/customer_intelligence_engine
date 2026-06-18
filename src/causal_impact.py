"""
Causal-impact module: estimate the true effect of an event (e.g. a promotion).

The idea behind CausalImpact: model what daily revenue *would have been* without
the event (the counterfactual), then attribute the gap during the event window to
the event. The DSI course uses Google's CausalImpact library (Bayesian structural
time series). Here we build a transparent, dependency-light version: fit a linear
model of daily revenue on a time trend + day-of-week using ONLY the pre-period,
project it through the event window, and measure actual minus predicted.

Swap in the `causalimpact` package for the production-grade Bayesian version; the
interface and interpretation are the same.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import FIGURES, MODELS
from src.data_prep import load_transactions


def run_causal_impact(event_start: str | None = None, event_days: int = 14) -> dict:
    tx = load_transactions()
    daily = (tx.set_index("InvoiceDate")["Revenue"].resample("D").sum()
               .rename("revenue").to_frame())
    daily = daily.asfreq("D", fill_value=0.0).reset_index()

    # default event window matches the synthetic promo (day 210); for real data,
    # pass the date of the campaign you want to evaluate.
    if event_start is None:
        event_start = (daily["InvoiceDate"].min() + pd.Timedelta(days=210)).strftime("%Y-%m-%d")
    ev_start = pd.Timestamp(event_start)
    ev_end = ev_start + pd.Timedelta(days=event_days)

    daily["t"] = (daily["InvoiceDate"] - daily["InvoiceDate"].min()).dt.days
    daily["dow"] = daily["InvoiceDate"].dt.dayofweek
    dow_dummies = pd.get_dummies(daily["dow"], prefix="dow")
    Xfull = pd.concat([daily[["t"]], dow_dummies], axis=1)

    pre = daily["InvoiceDate"] < ev_start
    model = LinearRegression().fit(Xfull[pre], daily.loc[pre, "revenue"])
    daily["counterfactual"] = model.predict(Xfull)

    window = (daily["InvoiceDate"] >= ev_start) & (daily["InvoiceDate"] < ev_end)
    actual = float(daily.loc[window, "revenue"].sum())
    expected = float(daily.loc[window, "counterfactual"].sum())
    lift = actual - expected

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily["InvoiceDate"], daily["revenue"], lw=0.9, label="actual")
    ax.plot(daily["InvoiceDate"], daily["counterfactual"], lw=1.2, ls="--",
            color="grey", label="counterfactual")
    ax.axvspan(ev_start, ev_end, color="#d97724", alpha=0.15, label="event window")
    ax.set_title("Causal impact of the event on daily revenue"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "causal_impact.png", dpi=110); plt.close(fig)

    result = {
        "event_window": [ev_start.strftime("%Y-%m-%d"), ev_end.strftime("%Y-%m-%d")],
        "actual_revenue": round(actual, 2),
        "expected_revenue": round(expected, 2),
        "estimated_lift": round(lift, 2),
        "relative_lift_pct": round(100 * lift / expected, 1) if expected else None,
    }
    pd.Series(result).to_json(MODELS / "causal_impact.json")
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(run_causal_impact(), indent=2))