"""Historical VaR, Expected Shortfall and assumption-explicit Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TailRisk:
    confidence: float
    var_loss: float
    expected_shortfall_loss: float
    unit: str
    method: str


def historical_var_es(returns: pd.Series, *, confidence: float = 0.95) -> TailRisk:
    if len(returns) < 20 or returns.isna().any() or not 0.5 < confidence < 1:
        raise ValueError("At least 20 complete returns and 0.5 < confidence < 1 are required")
    values = returns.astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Returns must be finite")
    losses = -values
    threshold = float(np.quantile(losses, confidence, method="higher"))
    tail = losses[losses >= threshold]
    if tail.size == 0:
        raise ValueError("No observations in empirical loss tail")
    expected_shortfall = float(tail.mean())
    if not np.isfinite([threshold, expected_shortfall]).all() or expected_shortfall < threshold:
        raise ValueError("Historical tail-risk estimates failed consistency checks")
    return TailRisk(confidence, threshold, expected_shortfall, "return_fraction", "historical")


def monte_carlo_normal(*, mean_return: float, volatility: float, horizon_sessions: int,
                       simulations: int = 20_000, confidence: float = 0.95,
                       seed: int | None = None) -> dict[str, float | str | int]:
    """Normal IID log-return increments; exponentiated to positive gross return."""
    values = np.asarray([mean_return, volatility], dtype=float)
    if not np.isfinite(values).all() or volatility < 0:
        raise ValueError("Mean must be finite and volatility finite/non-negative")
    if horizon_sessions < 1 or simulations < 100 or not 0.5 < confidence < 1:
        raise ValueError("Invalid horizon, simulation count, or confidence")
    rng = np.random.default_rng(seed)
    increments = rng.normal(mean_return, volatility, (simulations, horizon_sessions))
    cumulative = np.exp(increments.sum(axis=1)) - 1
    if not np.isfinite(cumulative).all():
        raise ValueError("Monte Carlo paths overflowed; reduce horizon or input scale")
    lower, median, upper = np.quantile(cumulative, [(1-confidence)/2, 0.5, 1-(1-confidence)/2])
    return {"distribution": "normal_iid_log_returns", "seed": seed,
            "simulations": simulations, "horizon_sessions": horizon_sessions,
            "median_return": float(median), "interval_lower": float(lower),
            "interval_upper": float(upper), "confidence": confidence}
